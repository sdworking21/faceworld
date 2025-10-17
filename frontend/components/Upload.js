// frontend/components/Upload.js
import React from "react";
import axios from "axios";

export default function Upload({ setResult }){
  const [file, setFile] = React.useState(null);
  const [domain, setDomain] = React.useState("celebrity");
  const [loading, setLoading] = React.useState(false);

  const sendPreview = async () => {
    if(!file) return alert("Choose file");
    const form = new FormData();
    form.append("file", file);
    setLoading(true);
    try {
      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/preview`, form, {
        headers: {"Content-Type":"multipart/form-data"},
      });
      setResult({ preview_url: `${process.env.NEXT_PUBLIC_API_URL}${res.data.preview_url}` });
    } catch(e){
      alert("Preview failed: " + (e.response?.data?.error || e.message));
    } finally { setLoading(false) }
  };

  const sendSearch = async () => {
    if(!file) return alert("Choose file");
    const form = new FormData();
    form.append("file", file);
    form.append("domain", domain);
    setLoading(true);
    try {
      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/search`, form, { headers: {"Content-Type":"multipart/form-data"} });
      setResult({ ...res.data });
    } catch(e){
      alert("Search failed: "+ (e.response?.data?.error || e.message));
    } finally { setLoading(false) }
  }

  const sendGenerate = async (style="cartoon") => {
    if(!file) return alert("Choose file");
    const form = new FormData();
    form.append("file", file);
    form.append("style", style);
    setLoading(true);
    try {
      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/generate`, form, { headers: {"Content-Type":"multipart/form-data"} , timeout: 120000});
      setResult({ ...res.data });
    } catch(e){
      alert("Generate failed: "+ (e.response?.data?.error || e.message));
    } finally { setLoading(false) }
  }

  return (
    <div className="bg-white/5 p-4 rounded-md">
      <input type="file" accept="image/*" onChange={(e)=>setFile(e.target.files[0])} />
      <div className="mt-4 flex gap-2">
        <select value={domain} onChange={e=>setDomain(e.target.value)} className="p-2 rounded bg-white/10">
          <option value="celebrity">Celebrity</option>
          <option value="cartoon">Cartoon</option>
          <option value="animal">Animal</option>
          <option value="all">All</option>
        </select>
        <button onClick={sendPreview} className="px-4 py-2 bg-indigo-600 rounded">Preview</button>
        <button onClick={sendSearch} className="px-4 py-2 bg-green-600 rounded">Find Similar</button>
        <button onClick={()=>sendGenerate("cartoon")} className="px-4 py-2 bg-pink-600 rounded">Generate Cartoon</button>
        <button onClick={()=>sendGenerate("anime")} className="px-4 py-2 bg-orange-600 rounded">Generate Anime</button>
      </div>
      {loading && <div className="mt-2">Processing...</div>}
    </div>
  )
}

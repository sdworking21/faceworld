// frontend/components/Results.js
import React from "react";

export default function Results({ result }){
  if(!result) return null;
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Preview or generated */}
      {result.preview_url && (
        <div className="bg-white/5 p-4 rounded">
          <h3 className="font-bold mb-2">Detected Face</h3>
          <img src={result.preview_url} alt="preview" className="max-w-full rounded" />
        </div>
      )}
      {result.generated_url && (
        <div className="bg-white/5 p-4 rounded">
          <h3 className="font-bold mb-2">Generated</h3>
          <img src={`${process.env.NEXT_PUBLIC_API_URL}${result.generated_url}`} alt="generated" className="max-w-full rounded" />
          <a className="mt-2 inline-block bg-indigo-600 px-3 py-1 rounded" href={`${process.env.NEXT_PUBLIC_API_URL}${result.generated_url}`} target="_blank" rel="noreferrer">Open</a>
        </div>
      )}
      {result.results && result.results.length>0 && (
        <div className="bg-white/5 p-4 rounded col-span-2">
          <h3 className="font-bold mb-2">Top Matches</h3>
          <div className="flex gap-3">
            {result.results.map((r, i)=>(
              <div key={i} className="bg-slate-800 p-2 rounded">
                <img src={`${process.env.NEXT_PUBLIC_STATIC_BASE || ""}${r.path}`} alt="match" className="w-32 h-32 object-cover rounded" />
                <div className="text-sm mt-1">Score: {r.score.toFixed(2)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

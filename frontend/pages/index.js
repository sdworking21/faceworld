// frontend/pages/index.js
import React from "react";
import Upload from "../components/Upload";
import Results from "../components/Results";

export default function Home(){
  const [result, setResult] = React.useState(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-indigo-900 text-white p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-4">FaceWorld — See Yourself as Cartoon, Anime, or Celebrity</h1>
        <Upload setResult={setResult} />
        <div className="mt-8">
          <Results result={result} />
        </div>
      </div>
    </div>
  )
}

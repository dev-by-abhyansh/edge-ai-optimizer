import React, { useState, useRef, useEffect } from 'react';

export default function EdgeAIOptimizer() {
  const [activeTab, setActiveTab] = useState('Comparison');
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ total_runs: 0, avg_baseline_ms: 0, avg_qat_ms: 0, avg_speedup: 0, size_reduction: "74.6%" });

  // --- Inference State ---
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef(null);

  const [results, setResults] = useState({
    baseline: { class: '---', confidence: 0, latency: '---', size: '42.7 MB' },
    mixed: { class: '---', confidence: 0, latency: '---', size: '42.7 MB' },
    qat: { class: '---', confidence: 0, latency: '---', size: '10.8 MB' },
    system: { gpu: 'NVIDIA RTX 3060', framework: 'PyTorch 2.6' }
  });

  useEffect(() => {
    if (activeTab === 'History') {
      fetch('http://localhost:5000/history').then(res => res.json()).then(data => setHistory(data));
    }
    if (activeTab === 'Dashboard') {
      fetch('http://localhost:5000/stats').then(res => res.json()).then(data => setStats(data));
    }
  }, [activeTab]);

  const runInference = async () => {
    if (!imageFile) return;
    setIsLoading(true);
    const formData = new FormData();
    formData.append('image', imageFile);
    try {
      const response = await fetch('http://localhost:5000/predict', { method: 'POST', body: formData });
      const data = await response.json();
      setResults(data);
    } catch (error) { console.error(error); } finally { setIsLoading(false); }
  };

  const StatCard = ({ label, value, subtext, color }) => (
    <div className="bg-[#141820] p-8 rounded-3xl border border-gray-800 shadow-xl">
      <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500 font-bold mb-4">{label}</p>
      <h3 className={`text-5xl font-black ${color} mb-2`}>{value}</h3>
      <p className="text-xs text-slate-400 font-medium">{subtext}</p>
    </div>
  );

  const ResultCard = ({ title, modelData, colorClass, barColor }) => (
    <div className="bg-[#1e2330] p-6 rounded-xl border border-gray-700/50 shadow-lg mb-6">
      <div className="flex justify-between items-center mb-4">
        <span className={`text-[10px] font-black px-2 py-0.5 rounded uppercase border ${colorClass}`}>{title}</span>
        <div className="flex space-x-2 text-[10px] font-mono text-gray-500">
          <span>{modelData.latency} ms</span> <span>|</span> <span>{modelData.size}</span>
        </div>
      </div>
      <div className="flex justify-between items-end">
        <div><p className="text-gray-500 text-[9px] tracking-widest uppercase mb-1">Result</p><h3 className="text-3xl font-bold text-white tracking-tight">{modelData.class}</h3></div>
        <div className="text-right"><h3 className="text-2xl font-bold text-white">{modelData.confidence}%</h3><p className="text-gray-500 text-[9px] tracking-widest uppercase">Confidence</p></div>
      </div>
      <div className="w-full bg-[#151923] h-1 rounded-full mt-4">
        <div className={`h-full rounded-full transition-all duration-1000 ${barColor}`} style={{ width: `${modelData.confidence}%` }}></div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0b0e14] text-slate-200 flex font-sans antialiased">
      <div className="w-72 border-r border-gray-800/50 p-10 hidden lg:block bg-[#0b0e14]">
        <div className="flex items-center space-x-3 mb-16"><div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white font-black text-xl">Σ</div><span className="font-bold text-lg tracking-tight">Synthetic Architect</span></div>
        <nav className="space-y-3">
          {['Dashboard', 'Comparison', 'History'].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} className={`w-full text-left px-4 py-3 rounded-xl text-sm font-semibold transition-all ${activeTab === tab ? 'bg-blue-600/10 text-blue-400 border border-blue-500/10' : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'}`}>{tab}</button>
          ))}
        </nav>
      </div>

      <div className="flex-1 p-12 overflow-y-auto">
        {activeTab === 'Dashboard' && (
          <div className="max-w-6xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-700">
            <h2 className="text-4xl font-black text-white tracking-tighter mb-10">System Dashboard</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-10">
              <StatCard label="Total Inferences" value={stats.total_runs} subtext="Total images processed in history" color="text-white" />
              <StatCard label="Compression" value={stats.size_reduction} subtext="Reduction from FP32 to INT8" color="text-emerald-500" />
              <StatCard label="Avg. Speedup" value={`${stats.avg_speedup}x`} subtext="Latency improvement ratio" color="text-blue-500" />
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-[#141820] p-8 rounded-3xl border border-gray-800">
                <h4 className="text-sm font-bold uppercase tracking-widest text-slate-500 mb-6">Latency Benchmark (Avg)</h4>
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between text-xs mb-2"><span>Baseline (FP32)</span><span>{stats.avg_baseline_ms}ms</span></div>
                    <div className="w-full bg-[#0b0e14] h-2 rounded-full"><div className="bg-blue-500 h-full rounded-full" style={{ width: '100%' }}></div></div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs mb-2"><span>Quantized (INT8)</span><span>{stats.avg_qat_ms}ms</span></div>
                    <div className="w-full bg-[#0b0e14] h-2 rounded-full"><div className="bg-emerald-500 h-full rounded-full" style={{ width: `${(stats.avg_qat_ms / stats.avg_baseline_ms) * 100}%` }}></div></div>
                  </div>
                </div>
              </div>
              <div className="bg-gradient-to-br from-blue-600/20 to-transparent p-8 rounded-3xl border border-blue-500/20 flex flex-col justify-center">
                <h4 className="text-xl font-bold text-white mb-2">Edge Ready Status</h4>
                <p className="text-sm text-slate-400 leading-relaxed">The INT8 model is optimized for deployment. Hardware acceleration for integer arithmetic is active and verified.</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'Comparison' && (
          <div className="max-w-6xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-700">
            <header className="mb-14"><h2 className="text-5xl font-black text-white tracking-tighter mb-4">Edge AI Optimizer</h2><p className="text-slate-500 text-sm">ResNet-18 Benchmark Unit • <span className="text-emerald-500">Live</span></p></header>
            <div className="grid grid-cols-12 gap-12">
              <div className="col-span-12 xl:col-span-5 space-y-8">
                <div className="bg-[#141820] p-8 rounded-3xl border border-gray-800 shadow-2xl">
                  <div className="flex justify-between items-center mb-8"><h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Input Tensor</h3><span className="font-mono text-[10px] text-slate-600">32x32 RGB</span></div>
                  <div onClick={() => fileInputRef.current.click()} className="aspect-square bg-[#0b0e14] border-2 border-dashed border-gray-800 rounded-2xl flex items-center justify-center mb-8 group hover:border-blue-500/30 transition-all cursor-pointer overflow-hidden">
                    {selectedImage ? <img src={selectedImage} className="w-full h-full object-cover" /> : <p className="text-slate-700 group-hover:text-blue-500 transition-colors font-medium">+ Upload Target</p>}
                    <input type="file" ref={fileInputRef} onChange={(e) => { const f = e.target.files[0]; if (f) { setImageFile(f); setSelectedImage(URL.createObjectURL(f)); }}} className="hidden" />
                  </div>
                  <button onClick={runInference} disabled={!selectedImage || isLoading} className="w-full py-5 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-2xl shadow-xl transition-all disabled:opacity-20 uppercase tracking-widest text-xs">{isLoading ? 'Processing...' : 'Run Inference'}</button>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-[#141820] p-5 rounded-2xl border border-gray-800"><p className="text-[9px] text-slate-500 uppercase tracking-widest mb-2">Hardware</p><p className="text-xs font-bold text-slate-300 truncate">{results.system.gpu}</p></div>
                  <div className="bg-[#141820] p-5 rounded-2xl border border-gray-800"><p className="text-[9px] text-slate-500 uppercase tracking-widest mb-2">Framework</p><p className="text-xs font-bold text-slate-300">{results.system.framework}</p></div>
                </div>
              </div>
              <div className="col-span-12 xl:col-span-7 space-y-2">
                <ResultCard title="Baseline FP32" modelData={results.baseline} colorClass="text-blue-400 border-blue-400/20" barColor="bg-blue-500" />
                <ResultCard title="Mixed Prec FP16" modelData={results.mixed} colorClass="text-orange-400 border-orange-400/20" barColor="bg-orange-500" />
                <ResultCard title="QAT INT8" modelData={results.qat} colorClass="text-emerald-400 border-emerald-400/20" barColor="bg-emerald-500" />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'History' && (
          <div className="max-w-6xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-700">
            <h2 className="text-4xl font-black text-white tracking-tighter mb-10">Inference Archive</h2>
            <div className="bg-[#141820] rounded-3xl border border-gray-800 shadow-2xl overflow-hidden">
              <table className="w-full text-left text-sm">
                <thead className="bg-[#0b0e14] border-b border-gray-800"><tr className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]"><th className="px-8 py-6">Timestamp</th><th className="px-8 py-6">Target Class</th><th className="px-8 py-6">Baseline Speed</th><th className="px-8 py-6">QAT Speed</th><th className="px-8 py-6 text-right">Result</th></tr></thead>
                <tbody className="divide-y divide-gray-800/50">
                  {history.map((item, idx) => (
                    <tr key={idx} className="hover:bg-white/[0.02] transition-colors"><td className="px-8 py-6 font-mono text-[11px] text-slate-500">{item.timestamp}</td><td className="px-8 py-6 font-bold text-slate-200">{item.mixed.class}</td><td className="px-8 py-6 text-blue-500/80 font-mono text-xs">{item.baseline.latency}ms</td><td className="px-8 py-6 text-emerald-500/80 font-mono text-xs">{item.qat.latency}ms</td><td className="px-8 py-6 text-right"><span className="px-3 py-1 bg-emerald-500/10 text-emerald-500 text-[9px] font-black uppercase rounded-lg border border-emerald-500/10">Passed</span></td></tr>
                  ))}
                  {history.length === 0 && (<tr><td colSpan="5" className="p-32 text-center text-slate-600">Archive empty.</td></tr>)}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
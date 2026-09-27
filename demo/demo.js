const demoData = fetch('/data.json').then(r => { if (!r.ok) throw new Error('演示数据加载失败'); return r.json(); });
async function demoFetch(path, options = {}) {
  const data = await demoData;
  if (path === '/api/library') return Response.json(data.library);
  if (path === '/api/search') {
    const body = typeof options.body === 'string' ? JSON.parse(options.body) : options.body;
    const item = data.cases.find(item => item.question === body?.question?.trim());
    if (!item) return Response.json({error:'demo_only', message:'此问题没有预制结果，请选择一个测试场景。'}, {status:422});
    if (item.error) return Response.json({error:item.error.code,message:item.error.message},{status:422});
    return Response.json(item.response);
  }
  return Response.json({message:'演示版不支持此操作'}, {status:404});
}

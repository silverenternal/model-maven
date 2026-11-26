-- Exercise the real renderer inside Neovim's terminal job, including VT input.
-- This is not a substitute for a human or a full physical-terminal walkthrough.
local root = assert(vim.env.MODEL_MAVEN_TUI_SOURCE, 'MODEL_MAVEN_TUI_SOURCE is required')
local renderer = assert(vim.env.MODEL_MAVEN_TUI_BIN, 'MODEL_MAVEN_TUI_BIN is required')
vim.opt.runtimepath:prepend(root)
vim.o.columns, vim.o.lines = 180, 45
local state = {closed=false, opts={project=vim.fn.getcwd()}}
local doc = {
  nodes={{id='input',kind='input',inputs={},outputs={'value'}},
    {id='output',kind='output',inputs={'value'},outputs={}}},
  definitions={}, edges={{from={node='input',port='value'},to={node='output',port='value'}}},
}
local view = assert(require('model_maven.model_live_preview').open(state,doc,'sample',renderer))
-- Keep the terminal buffer available if the child exits, so failures include
-- its actual error instead of being hidden by the normal fallback cleanup.
local function contents()
  if not vim.api.nvim_buf_is_valid(view.buf) then error('renderer closed unexpectedly') end
  return table.concat(vim.api.nvim_buf_get_lines(view.buf,0,-1,false),'\n')
end
local function camera()
  return contents():match('CAM%s+(%d+)°/([%+%-]%d+)°')
end
if not vim.wait(10000,function() return camera() ~= nil end,50) then
  local status = vim.fn.jobwait({view.job}, 0)[1]
  error(('3D view did not appear (job status %s, closed %s):\n%s\nMessages:\n%s')
    :format(status, tostring(view.closed), contents(), vim.fn.execute('messages')))
end
local before = assert(camera())
vim.api.nvim_chan_send(view.job,'\27[C')
assert(vim.wait(5000,function() local current=camera(); return current and current ~= before end,50), 'camera did not rotate: '..contents())
local after_key = assert(camera())
local function send(bytes)
  vim.api.nvim_chan_send(view.job, bytes)
  vim.wait(120)
end
send('\27[<0;35;10M')
send('\27[<32;40;11M')
send('\27[<32;45;12M')
send('\27[<0;45;12m')
assert(vim.wait(5000,function() local current=camera(); return current and current ~= after_key end,50), 'mouse drag did not rotate: '..contents())
local after_mouse = assert(camera())
doc.nodes[#doc.nodes+1] = {id='extra',kind='input',inputs={},outputs={'value'}}
assert(view.update(doc,'draft'), 'cannot update preview')
assert(vim.wait(5000,function() return contents():find('/3',1,true) ~= nil end,50), 'draft update was not rendered: '..contents())
assert(camera() == after_mouse, 'draft update reset camera')
print('live 3D: keyboard orbit '..before..' -> '..after_key..'; mouse -> '..after_mouse..'; draft has 3 nodes')
state.closed = true
view.close()
vim.cmd('qa!')

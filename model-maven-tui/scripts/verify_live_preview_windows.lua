-- Exercise the renderer, then drive its floating window in an attached UI.
vim.env.MODEL_MAVEN_PROBE_EMBEDDED = '1'
local finished = false
local function finish(ok, result)
  if finished then return end
  finished = true
  local f = assert(io.open(vim.env.MM_PROBE_OUT, 'w'))
  f:write(vim.json.encode({ok=ok, result=ok and result or nil, error=not ok and tostring(result) or nil})); f:close()
  vim.cmd('qa!')
end
local ok, renderer_result = pcall(dofile, 'model-maven-tui/scripts/verify_live_preview_terminal.lua')
if not ok then finish(false, renderer_result); return end
local state={closed=false,opts={project=vim.fn.getcwd()}}
local doc={nodes={{id='input',kind='input',inputs={},outputs={'value'}}},definitions={},edges={}}
local preview=require('model_maven.model_live_preview')
local view=assert(preview.open(state,doc,'draft',vim.env.MODEL_MAVEN_TUI_BIN))
state.model_preview=view
vim.o.mouse='a'
local before={row=view.row,col=view.col,width=view.width}
local row,col=view.row+1,view.col+5
local function later(delay, fn)
  vim.defer_fn(function()
    if finished then return end
    local success, err=pcall(fn)
    if not success then finish(false,err) end
  end,delay)
end
later(500,function() vim.api.nvim_input_mouse('left','press','',0,row,col) end)
later(700,function() vim.api.nvim_input_mouse('left','drag','',0,row+2,col-5) end)
later(900,function() vim.api.nvim_input_mouse('left','release','',0,row+2,col-5) end)
later(1200,function()
  assert(view.row ~= before.row or view.col ~= before.col, 'preview window did not move')
  vim.api.nvim_set_current_win(view.win)
  vim.api.nvim_input('+')
end)
later(1500,function()
  assert(view.width > before.width, 'preview window did not resize')
  vim.api.nvim_input('q')
end)
later(1800,function()
  assert(view.closed and state.model_preview == nil, 'preview did not close')
  state.model_preview_hidden=false
  local reopened=assert(preview.open(state,doc,'draft',vim.env.MODEL_MAVEN_TUI_BIN))
  state.model_preview=reopened
  assert(vim.wait(5000,function()
    return vim.api.nvim_buf_is_valid(reopened.buf)
      and table.concat(vim.api.nvim_buf_get_lines(reopened.buf,0,-1,false),'\n'):find('CAM',1,true)
  end,50), 'reopened preview did not render')
  local result={renderer=renderer_result,window={moved=true,resized=true,closed=true,reopened=true}}
  reopened.close()
  finish(true,result)
end)

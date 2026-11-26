local path = 'model-maven-tui/scripts/verify_live_preview_terminal.lua'
local script = assert(io.open(path)):read('*a'):gsub("vim.cmd%('qa!'%)", '')
local ok, err = pcall(assert(loadstring(script, '@'..path)))
local f = assert(io.open(vim.env.MM_PROBE_OUT, 'w'))
f:write(vim.json.encode({ok=ok, error=tostring(err)})); f:close()
vim.cmd('qa!')

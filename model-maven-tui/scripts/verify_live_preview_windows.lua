-- Run inside a Neovim instance attached to a Windows pseudo-console.
vim.env.MODEL_MAVEN_PROBE_EMBEDDED = '1'
local ok, result = pcall(dofile, 'model-maven-tui/scripts/verify_live_preview_terminal.lua')
local f = assert(io.open(vim.env.MM_PROBE_OUT, 'w'))
f:write(vim.json.encode({ok=ok, result=ok and result or nil, error=not ok and tostring(result) or nil})); f:close()
vim.cmd('qa!')

vim.o.columns=180; vim.o.lines=45
local buf=vim.api.nvim_create_buf(false,true)
local win=vim.api.nvim_open_win(buf,true,{relative='editor',row=1,col=1,width=90,height=30})
local job=vim.fn.jobstart({'cmd.exe','/c','echo HELLO_CONPTY && timeout /t 3'}, {term=true})
assert(job > 0)
vim.wait(2000,function() return table.concat(vim.api.nvim_buf_get_lines(buf,0,-1,false),'\n'):find('HELLO_CONPTY',1,true) ~= nil end,50)
print('status='..vim.fn.jobwait({job},0)[1]..' lines='..vim.inspect(vim.api.nvim_buf_get_lines(buf,0,-1,false)))
vim.cmd('qa!')

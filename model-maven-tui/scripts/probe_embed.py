import os,time,pynvim
n=pynvim.attach('child',argv=[os.environ['MM_CI_NVIM'],'--embed','-u','NONE','-n'])
n.ui_attach(180,45,rgb=True)
n.exec_lua("""
local b=vim.api.nvim_create_buf(false,true)
vim.api.nvim_open_win(b,true,{relative='editor',row=1,col=1,width=90,height=30})
_G.probe_buf=b
_G.probe_job=vim.fn.jobstart({'cmd.exe','/c','echo HELLO_CONPTY && ping -n 5 127.0.0.1'}, {term=true})
""")
time.sleep(2)
print(n.exec_lua("return {job=_G.probe_job,status=vim.fn.jobwait({_G.probe_job},0)[1],lines=vim.api.nvim_buf_get_lines(_G.probe_buf,0,-1,false)}"))
n.close()

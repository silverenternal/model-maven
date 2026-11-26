-- Durable marker for a foreground operation whose receipt may be lost.
-- It records no arguments or environment values.
local M = {}
local uv = vim.uv

local function path(project, create)
  local dir = project .. '/.model-maven'
  local stat, err, code = uv.fs_lstat(dir)
  if not stat and code ~= 'ENOENT' then return nil, err end
  if not stat and create then
    local ok, mkdir_err = uv.fs_mkdir(dir, 448)
    if not ok and not uv.fs_lstat(dir) then return nil, mkdir_err end
    stat = uv.fs_lstat(dir)
  end
  if not stat then return nil, nil end
  if stat.type ~= 'directory' then return nil, 'Project metadata is not a real directory.' end
  return dir .. '/tui-foreground-pending.json', nil, dir
end

local function sync_directory(dir)
  if vim.fn.has('win32') == 1 then return true end
  local fd, err = uv.fs_open(dir, 'r', 0)
  if not fd then return nil, err end
  local ok, sync_err = uv.fs_fsync(fd)
  uv.fs_close(fd)
  return ok, sync_err
end

function M.read(project)
  local name, err = path(project, false)
  if not name then return nil, err end
  local stat, stat_err, code = uv.fs_lstat(name)
  if not stat then
    if code == 'ENOENT' then return nil, nil end
    return nil, stat_err
  end
  if stat.type ~= 'file' or stat.size > 4096 then return nil, 'Pending foreground marker is invalid.' end
  local fd, open_err = uv.fs_open(name, 'r', 0)
  if not fd then return nil, open_err end
  local raw, read_err = uv.fs_read(fd, 4097, 0)
  uv.fs_close(fd)
  if not raw then return nil, read_err end
  local ok, value = pcall(vim.json.decode, raw)
  if not ok or type(value) ~= 'table' or value.version ~= 1 or type(value.id) ~= 'string'
      or not value.id:match('^[0-9a-f]+$') or #value.id ~= 40 or type(value.title) ~= 'string'
      or #value.title > 512 then return nil, 'Pending foreground marker is invalid.' end
  return value
end

function M.create(project, title)
  local name, err, dir = path(project, true)
  if not name then return nil, err end
  local bytes, random_err = uv.random(20)
  if not bytes then return nil, random_err end
  local id = bytes:gsub('.', function(c) return string.format('%02x', string.byte(c)) end)
  local value = { version = 1, id = id, title = vim.fn.strcharpart(tostring(title):gsub('[%c]', ' '), 0, 128),
    host = vim.fn.strcharpart(vim.fn.hostname():gsub('[%c]', ' '), 0, 128),
    editor_pid = uv.os_getpid() }
  local raw = vim.json.encode(value)
  local fd, open_err = uv.fs_open(name, 'wx', 384)
  if not fd then return nil, open_err end
  local written, write_err = uv.fs_write(fd, raw, 0)
  local ok, sync_err
  if written == #raw then ok, sync_err = uv.fs_fsync(fd) end
  uv.fs_close(fd)
  if not ok then return nil, write_err or sync_err or 'Cannot write pending foreground marker.' end
  local synced, dir_err = sync_directory(dir)
  if not synced then return nil, dir_err end
  return value
end

function M.clear(project, expected)
  local current, err = M.read(project)
  if not current then return nil, err or 'Pending foreground marker is missing.' end
  if not expected or current.id ~= expected.id then return nil, 'Pending foreground marker changed.' end
  local name, path_err, dir = path(project, false)
  if not name then return nil, path_err end
  local ok, unlink_err = uv.fs_unlink(name)
  if not ok then return nil, unlink_err end
  return sync_directory(dir)
end

return M

-- Compatible with the standalone workbench's durable request markers.
local M = {}
local uv = vim.uv
local locale = require('model_maven.locale')

local function message(en, zh, tw)
  return locale.text(en, zh, tw)
end

local function valid(id)
  return type(id) == 'string' and #id >= 16 and #id <= 64
    and id:match('^tui%-[a-zA-Z0-9%-]+$') ~= nil
end

local function directory(project, create)
  local metadata = project .. '/.model-maven'
  local root = metadata .. '/tui-pending-starts'
  local meta, meta_err, meta_code = uv.fs_lstat(metadata)
  if not meta and meta_code ~= 'ENOENT' then return nil, meta_err end
  if not meta and create then
    local ok, err = uv.fs_mkdir(metadata, 448)
    if not ok then
      meta = uv.fs_lstat(metadata)
      if not meta then return nil, err end
    end
    meta = meta or uv.fs_lstat(metadata)
  end
  if not meta then return nil, nil end
  if meta.type ~= 'directory' then
    return nil, message('Project metadata directory is unavailable or is a link.', '项目元数据目录不可用或是链接。', '專案中繼資料目錄無法使用或是連結。')
  end
  local stat, stat_err, stat_code = uv.fs_lstat(root)
  if not stat and stat_code ~= 'ENOENT' then return nil, stat_err end
  if not stat and create then
    local ok, err = uv.fs_mkdir(root, 448)
    if not ok and not uv.fs_lstat(root) then return nil, err end
    stat = uv.fs_lstat(root)
  end
  if not stat then return nil, nil end
  if stat.type ~= 'directory' then
    return nil, message('Pending request path is not a real directory.', '待确认请求路径不是实际目录。', '待確認要求路徑不是實際目錄。')
  end
  return root
end

local function sync_dir(path)
  if vim.fn.has('win32') == 1 then return true end
  local fd, err = uv.fs_open(path, 'r', 0)
  if not fd then return nil, err end
  local ok, sync_err = uv.fs_fsync(fd)
  uv.fs_close(fd)
  return ok, sync_err
end

function M.list(project)
  local root, err = directory(project, false)
  if not root then
    if err then return nil, err end
    return {}
  end
  local scan, scan_err = uv.fs_scandir(root)
  if not scan then return nil, scan_err end
  local items = {}
  while true do
    local name = uv.fs_scandir_next(scan)
    if not name then break end
    -- This directory is reserved for request markers and the admission
    -- lock. Do not let an unknown or damaged marker name look like an empty
    -- request queue and admit another start.
    if name ~= '.admission-lock' then
      if not valid(name) then
        return nil, message('Invalid pending request marker.', '待确认请求标记无效。', '待確認要求標記無效。')
      end
      local stat = uv.fs_lstat(root .. '/' .. name)
      if not stat or stat.type ~= 'file' then
        return nil, message('Invalid pending request marker.', '待确认请求标记无效。', '待確認要求標記無效。')
      end
      items[#items + 1] = name
      if #items > 256 then
        return nil, message('Too many pending request markers.', '待确认请求标记过多。', '待確認要求標記過多。')
      end
    end
  end
  table.sort(items)
  return items
end

function M.create(project)
  local root, err = directory(project, true)
  if not root then return nil, err end
  -- The metadata directory may have been created concurrently; sync the
  -- parent in either case before relying on a durable request marker.
  local parent_synced, parent_error = sync_dir(project)
  if not parent_synced then return nil, parent_error end
  local synced, sync_err = sync_dir(project .. '/.model-maven')
  if not synced then return nil, sync_err end
  local lock = root .. '/.admission-lock'
  local locked, lock_err = uv.fs_mkdir(lock, 448)
  if not locked then
    return nil, message('Training admission is busy or needs inspection: ', '训练请求登记正在进行或需要检查：', '訓練要求登記正在進行或需要檢查：') .. tostring(lock_err)
  end
  local function release()
    local ok, release_err = uv.fs_rmdir(lock)
    if not ok then return nil, release_err end
    return sync_dir(root)
  end
  local existing, list_err = M.list(project)
  if not existing or #existing > 0 then
    local ok, release_err = release()
    return nil, not ok and release_err or list_err
      or message('An earlier training request is unconfirmed.', '先前的训练请求尚未确认。', '先前的訓練要求尚未確認。')
  end
  local bytes, random_err = uv.random(20)
  if not bytes or #bytes ~= 20 then
    release()
    return nil, random_err or message('Secure randomness unavailable.', '无法获取安全随机数。', '無法取得安全亂數。')
  end
  local id = 'tui-' .. bytes:gsub('.', function(c) return string.format('%02x', string.byte(c)) end)
  local path = root .. '/' .. id
  local fd, open_err = uv.fs_open(path, 'wx', 384)
  if not fd then release(); return nil, open_err end
  local ok
  ok, sync_err = uv.fs_fsync(fd)
  uv.fs_close(fd)
  if ok then ok, sync_err = sync_dir(root) end
  if not ok then uv.fs_unlink(path); release(); return nil, sync_err end
  local released, release_err = release()
  if not released then
    return nil, message('Request marker exists but admission lock could not be released: ', '请求标记已创建，但无法释放登记锁：', '要求標記已建立，但無法釋放登記鎖：') .. tostring(release_err)
  end
  return id
end

function M.lock_status(project)
  local root, err = directory(project, false)
  if not root then
    if err then return nil, err end
    return false
  end
  local path = root .. '/.admission-lock'
  local stat, stat_err, code = uv.fs_lstat(path)
  if not stat then
    if code == 'ENOENT' then return false end
    return nil, stat_err
  end
  if stat.type ~= 'directory' then
    return nil, message('Admission lock is not a real directory.', '登记锁不是实际目录。', '登記鎖不是實際目錄。')
  end
  return path
end

function M.clear_lock(project)
  local path, err = M.lock_status(project)
  if not path then return nil, err or message('Admission lock is missing.', '登记锁不存在。', '登記鎖不存在。') end
  local ok, remove_err = uv.fs_rmdir(path)
  if not ok then return nil, remove_err end
  local root = directory(project, false)
  return sync_dir(root)
end

function M.clear(project, id)
  if not valid(id) then return nil, message('Invalid request identity.', '请求身份无效。', '要求身分無效。') end
  local root, err = directory(project, false)
  if not root then return nil, err or message('Pending directory missing.', '待确认请求目录不存在。', '待確認要求目錄不存在。') end
  local path = root .. '/' .. id
  local stat = uv.fs_lstat(path)
  if not stat or stat.type ~= 'file' then
    return nil, message('Pending marker missing or invalid.', '待确认请求标记不存在或无效。', '待確認要求標記不存在或無效。')
  end
  local ok, remove_err = uv.fs_unlink(path)
  if not ok then return nil, remove_err end
  return sync_dir(root)
end

return M

local json = require 'cjson'

function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function parse_info_file(info_file_path)
    local info_file = io.open(info_file_path, "rb")
    local info_file_content = info_file:read("*all")
    info_file:close()
    local film_info = json.decode(info_file_content)
    return film_info
end


function get_frame_size(films)
    return films[1].frames[1]:size()
end

-- updates current output line
local last_str = ""
function update_output(str)
    io.write(('\b \b'):rep(#last_str))
    io.write(str)                     
    io.flush()
    last_str = str
end

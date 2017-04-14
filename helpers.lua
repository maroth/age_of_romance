function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

-- updates current output line
local last_str = ""
function update_output(str)
    io.write(('\b \b'):rep(#last_str))
    io.write(str)                     
    io.flush()
    last_str = str
end

function count_frames(frame_files)
    local number_of_frames = 0
    for _, frame in ipairs(frame_files) do
        number_of_frames = number_of_frames + 1
    end
    return number_of_frames
end

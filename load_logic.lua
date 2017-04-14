local json = require 'cjson'

function get_frame_size(frame_dir)
    for film_dir in lfs.dir(frame_dir) do    
        local info_file_path = frame_dir .. "/" .. film_dir .. "/info.json"
        if file_exists(info_file_path) then
            for frame_file in lfs.dir(frame_dir .. "/" .. film_dir) do
                if (string.ends(frame_file, ".png")) then
                    local image_file_name = frame_dir .. film_dir .. "/" .. frame_file
                    return image.load(image_file_name, 3, 'double'):size()
                end
            end
        end
    end
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

function build_frame_set(frame_dir)
    local index = 0
    frame_files = {}
    frame_films = {}
    for film_dir in lfs.dir(frame_dir) do    
        local info_file_path = frame_dir .. "/" .. film_dir .. "/info.json"
        if file_exists(info_file_path) then
            local film  = parse_info_file(info_file_path)
            film.normalized_date = normalize_date(parse_date(film.date))
            for frame_file in lfs.dir(frame_dir .. "/" .. film_dir) do
                if (string.ends(frame_file, ".png")) then
                    index = index + 1
                    update_output("loading frame : " .. index)
                    frame_files[index] = frame_dir .. film_dir .. "/" .. frame_file
                    frame_films[index] = film
                end
            end
        end
    end
    print("")
    return frame_files, frame_films
end




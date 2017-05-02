local json = require 'cjson'

function get_frame_size(frame_dir, params)
    for film_dir in lfs.dir(frame_dir) do    
        local info_file_path = frame_dir .. "/" .. film_dir .. "/info.json"
        if file_exists(info_file_path) then
            for frame_file in lfs.dir(frame_dir .. "/" .. film_dir) do
                if (string.ends(frame_file, ".png")) then
                    local image_file_name = frame_dir .. film_dir .. "/" .. frame_file
                    return image.load(image_file_name, params.channels, 'double'):size()
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

function count_frames_in_dir(dir)
    local total_frames = 0
    for frame_file in lfs.dir(dir) do
        if (string.ends(frame_file, ".png")) then
            total_frames = total_frames + 1
        end
    end
    return total_frames
end

function frame_limits(total_frames, max_frames_per_directory)
    local from_frame = 1
    local to_frame = total_frames
    if max_frames_per_directory then
        from_frame = (total_frames / 2) - math.floor((max_frames_per_directory / 2))
        to_frame = (total_frames / 2) + math.ceil((max_frames_per_directory / 2))
    end
    return from_frame, to_frame
end

function load_films(frame_dir, max_frames_per_directory, number_of_bins)
    films = {}
    for film_dir in lfs.dir(frame_dir) do    
        local info_file_path = frame_dir .. film_dir .. "/info.json"
        if file_exists(info_file_path) then
            local film  = parse_info_file(info_file_path)

            local total_frames = count_frames_in_dir(frame_dir .. film_dir)
            local from_frame, to_frame = frame_limits(total_frames, max_frames_per_directory)

            film.normalized_date = normalize_date(parse_date(film.date))
            film.bin_vector = create_probability_vector(film.normalized_date[1], number_of_bins)
            film.bin = get_bin(film.normalized_date[1], number_of_bins)
            film.frames = {}
            local frame_count = 0
            for frame_file in lfs.dir(frame_dir .. film_dir) do
                if (string.ends(frame_file, ".png")) then
                    frames_count = frames_count + 1
                    if frames_count >= from_frame and frames_count <= to_frame then 
                        local frame_file_dir = frame_dir .. film_dir .. "/" .. frame_file
                        local frame_id = string.gsub(frame_file, "frame", "")
                        frame_id = string.gsub(frame_id, ".png", "")
                        film.frames[tonumber(frame_id)] = frame_file_dir
                    end
                end
            end
            table.insert(films, film)
        end
    end
    return films
end

function build_frame_set(frame_dir, max_frames_per_directory, number_of_bins)
    local index = 0
    frame_files = {}
    frame_films = {}
    for film_dir in lfs.dir(frame_dir) do    
        local info_file_path = frame_dir .. "/" .. film_dir .. "/info.json"
        if file_exists(info_file_path) then
            local film  = parse_info_file(info_file_path)

            local total_frames = count_frames_in_dir(frame_dir .. film_dir)
            local from_frame, to_frame = frame_limits(total_frames, max_frames_per_directory)

            film.normalized_date = normalize_date(parse_date(film.date))
            film.bin_vector = create_probability_vector(film.normalized_date[1], number_of_bins)
            film.bin = get_bin(film.normalized_date[1], number_of_bins)
            local frames_count = 0
            for frame_file in lfs.dir(frame_dir .. "/" .. film_dir) do
                if (string.ends(frame_file, ".png")) then
                    frames_count = frames_count + 1
                    if frames_count >= from_frame and frames_count <= to_frame then 
                        index = index + 1
                        if index % 1000 == 0 then
                            update_output("loading frame: " .. index)
                        end
                        frame_files[index] = frame_dir .. film_dir .. "/" .. frame_file
                        frame_films[index] = film
                    end
                end
            end
        end
    end
    return frame_files, frame_films
end




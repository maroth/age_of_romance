local json = require 'cjson'
require 'helpers'
require 'lfs'
require 'date_logic'
require 'image'

function get_frame_size(frame_dir, params)
    for film_dir in lfs.dir(frame_dir) do    
        local info_file_path = frame_dir .. "/" .. film_dir .. "/info.json"
        if file_exists(info_file_path) then
            for frame_file in lfs.dir(frame_dir .. "/" .. film_dir) do
                if (string.ends(frame_file, ".png")) then
                    local image_file_name = frame_dir .. film_dir .. "/" .. frame_file
                    local img = image.load(image_file_name, params.channels, 'double')
		    return get_color_distribution(img):size()
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
            local frames_count = 0
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

function all_cspaces()
    number_of_bins = 250
    save_cspaces("cspaces/test_data_" .. number_of_bins .. ".bin", "cspaces/test_labels_" .. number_of_bins .. ".bin", "frames_211x176_distributed/test/", number_of_bins)
    save_cspaces("cspaces/train_data_" .. number_of_bins .. ".bin", "cspaces/train_labels_" .. number_of_bins .. ".bin", "frames_211x176_distributed/train/", number_of_bins)
    save_cspaces("cspaces/validate_data_" .. number_of_bins .. ".bin", "cspaces/validate_labels_" .. number_of_bins .. ".bin", "frames_211x176_distributed/validate/", number_of_bins)
    
    save_cspaces("cspaces/continuous_test_data_" .. number_of_bins .. ".bin", "cspaces/continuous_test_labels_" .. number_of_bins .. ".bin", "frames_211x176_continuous/test/", number_of_bins)
    save_cspaces("cspaces/continuous_train_data_" .. number_of_bins .. ".bin", "cspaces/continuous_train_labels_" .. number_of_bins .. ".bin", "frames_211x176_continuous/train/", number_of_bins)
    save_cspaces("cspaces/continuous_validate_data_" .. number_of_bins .. ".bin", "cspaces/continuous_validate_labels_" .. number_of_bins .. ".bin", "frames_211x176_continuous/validate/", number_of_bins)
    
    save_cspaces("cspaces/separate_test_data_" .. number_of_bins .. ".bin", "cspaces/separate_test_labels_" .. number_of_bins .. ".bin", "frames_211x176_separate/test/", number_of_bins)
    save_cspaces("cspaces/separate_train_data_" .. number_of_bins .. ".bin", "cspaces/separate_train_labels_" .. number_of_bins .. ".bin", "frames_211x176_separate/train/", number_of_bins)
    save_cspaces("cspaces/separate_validate_data_" .. number_of_bins .. ".bin", "cspaces/separate_validate_labels_" .. number_of_bins .. ".bin", "frames_211x176_separate/validate/", number_of_bins)
end

function save_cspaces(file_cspaces, file_films, frame_dir, number_of_bins)
    local frame_films, frame_cspaces = build_frame_set(frame_dir, nil, number_of_bins)
    local frame_films_tensor = torch.CudaTensor(#frame_films)
    for i = 1, #frame_films do
        frame_films_tensor[i] = frame_films[i].normalized_date
    end
    local frame_cspaces_tensor = torch.CudaTensor(#frame_cspaces, 3, 176)
    for i = 1, #frame_cspaces do
        frame_cspaces_tensor[i] = frame_cspaces[i]
    end
    torch.save(file_films, frame_films_tensor)
    torch.save(file_cspaces, frame_cspaces_tensor)
end

function build_frame_set(frame_dir, max_frames_per_directory, number_of_bins)
    local index = 0
    frame_files = {}
    frame_films = {}
    frame_cspaces = {}
    for film_dir in lfs.dir(frame_dir) do    
        local info_file_path = frame_dir .. "/" .. film_dir .. "/info.json"
        if file_exists(info_file_path) then
            local film  = parse_info_file(info_file_path)

            local total_frames = count_frames_in_dir(frame_dir .. film_dir)
            local from_frame, to_frame = frame_limits(total_frames, max_frames_per_directory)

            film.normalized_date = normalize_date(parse_date(film.date))
            

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
                        frame_cspaces[index] = get_color_distribution(image.load(frame_files[index], 3, 'double'))
                    end
                end
            end
        end
    end
    return frame_films, frame_cspaces
end




require 'image'
require 'nn'
require 'lfs'

local json = require 'cjson'

train_frame_dir = "/mnt/e/age_of_romance/mini_frames/"
test_frame_dir = "/mnt/e/age_of_romance/mini_frames/"

zero_date = os.time({year = 1960, month = 1, day = 1})
one_date = os.time({year = 2010, month = 1, day = 1})

learning_rate = 0.001
epochs = 5

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

function parse_date(date_to_convert)
    local pattern = "(%d+)-(%d+)-(%d+)"
    local runyear, runmonth, runday = date_to_convert:match(pattern)
    local time_stamp = os.time({year = runyear, month = runmonth, day = runday})
    return time_stamp
end

function normalize_date(date_to_normalize)
    if date_to_normalize > one_date then
        print("ERROR: movie too new")
    end 
    if date_to_normalize < zero_date then
        print("ERROR: movie too old")
    end 
    local normalized_date = (date_to_normalize - zero_date) / (one_date - zero_date)
    local normalized_date_tensor = torch.DoubleTensor(1)
    normalized_date_tensor[1] = normalized_date
    return normalized_date_tensor
end

function denormalize_date(date_to_denormalize)
    denormalized_date = date_to_denormalize[1] * (one_date - zero_date) + zero_date
    date_table = os.date("*t", denormalized_date)
    date_string = date_table.year .. "-" .. date_table.month .. "-" .. date_table.day
    return date_string
end

function get_frame_size(films)
    return films[1].frames[1]:size()
end

function load_films(frame_dir)
    films = {}
    for film_dir in lfs.dir(frame_dir) do    
        local info_file_path = frame_dir .. "/" .. film_dir .. "/info.json"
        if file_exists(info_file_path) then
            local film  = parse_info_file(info_file_path)
            film.normalized_date = normalize_date(parse_date(film.date))
            film.frames = {}
            for frame_file in lfs.dir(frame_dir .. "/" .. film_dir) do
                if (string.ends(frame_file, ".png")) then
                    local frame_file_dir = frame_dir .. "/" .. film_dir .. "/" .. frame_file
                    local frame_id = string.gsub(frame_file, "frame", "")
                    frame_id = string.gsub(frame_id, ".png", "")
                    film.frames[tonumber(frame_id)] = image.load(frame_file_dir, 3, 'double')
                end
            end
            table.insert(films, film)
        end
    end
    print ("films loaded from " .. frame_dir)
    return films
end

function build_shuffled_frame_set(films)
    local number_of_frames = 0
    for film_index, film in ipairs(films) do
        for frame_index, frame in ipairs(film.frames) do
            number_of_frames = number_of_frames + 1
        end
    end

    local frame_size = get_frame_size(films)
    frames = torch.DoubleTensor(number_of_frames, frame_size[1], frame_size[2], frame_size[3])
    dates = torch.DoubleTensor(number_of_frames)
    index = 0
    for film_index, film in ipairs(films) do
        for frame_index, frame in ipairs(film.frames) do
            index = index + 1
            frames[index] = frame
            dates[index] = film.normalized_date
        end
    end

    randomized_indexes = torch.randperm(number_of_frames):long()
    shuffled_frames = frames:index(1, randomized_indexes)
    shuffled_dates = dates:index(1, randomized_indexes)
    return shuffled_frames, shuffled_dates
end

function train(films, learning_rate, epochs)
    local frame_size = get_frame_size(films)
    local input_size = frame_size[1] * frame_size[2] * frame_size[3]
    local neural_network = nn.Sequential()
    neural_network:add(nn.Linear(input_size, 1))
    neural_network:add(nn.Sigmoid())
    local criterion = nn.MSECriterion()

    for epoch_index = 1, epochs do
        local err_sum = 0
        frames, dates = build_shuffled_frame_set(films)
        for index = 1, dates:size(1) do
            frame = frames[index]
            true_date = torch.DoubleTensor(1)
            true_date[1] = dates[index]
            local reshaped_frame = torch.reshape(frame, input_size)
            local prediction = neural_network:forward(reshaped_frame)
            neural_network:zeroGradParameters()
            local err = criterion:forward(prediction, true_date)
            err_sum = err_sum + err
            local grad_criterion = criterion:backward(prediction, true_date)
            neural_network:backward(reshaped_frame, grad_criterion)
            neural_network:updateParameters(learning_rate)
        end
        print("error rate after epoch " .. epoch_index .. ": " .. err_sum / dates:size(1))
    end
    return neural_network
end

function test(neural_network, films)
    local frame_size = get_frame_size(films)
    local input_size = frame_size[1] * frame_size[2] * frame_size[3]
    for film_index, film in ipairs(films) do
        local frame_sum = 0
        local frame_count = 0
        for frame_index, frame in ipairs(film.frames) do
            frame_count = frame_count + 1
            local reshaped_frame = torch.reshape(frame, input_size)
            local prediction = neural_network:forward(reshaped_frame)
            frame_sum = frame_sum + prediction
        end
        
        local average_prediction = frame_sum / frame_count
        local denormalized_prediction = denormalize_date(average_prediction)        
        print(film.title)
        print("actual date: " .. film.date, "predicted date: " .. denormalized_prediction)
    end
end

train_films = load_films(train_frame_dir)
neural_network = train(films, learning_rate, epochs)

test_films = load_films(test_frame_dir)
test(neural_network, films)

require 'cunn'

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

function get_remaining_time(starting_time, fraction_done)
    local elapsed_seconds = os.time() - starting_time
    local estimated_total_seconds = elapsed_seconds / fraction_done
    local estimated_remaining_seconds = estimated_total_seconds - elapsed_seconds
    local formatted_time = string.format("%.2d:%.2d:%.2d", estimated_remaining_seconds / (60*60), estimated_remaining_seconds / 60 % 60, estimated_remaining_seconds % 60)
    return formatted_time
end

function minibatch_summary(minibatch_index, number_of_minibatches, epoch_index, epochs, starting_time, err)
    local fraction_done  = ((epoch_index - 1) * number_of_minibatches + (minibatch_index - 1)) / (epochs * number_of_minibatches)
    local estimated_remaining_time = get_remaining_time(starting_time, fraction_done)
    message = ("[" .. epoch_index .. "/" .. epochs .. "] " .. " minibatch " .. minibatch_index .. " of " .. number_of_minibatches .. ", error rate: " .. string.format("%.5f", err) .. " (" .. string.format("%.3f", fraction_done * 100) .. "%, remaining time: " .. estimated_remaining_time .. ")")
    return message
end

function minibatch_detail(minibatch_size, prediction, minibatch_dates, err)
    local message = ""
    for i = 1, minibatch_size do
        local local_prediction = 0
        if (prediction:size(1) > 1) then
            local_prediction = prediction[i][1]
        else
            local_prediction = prediction[i]
        end
        message = message .. "\nprediction: " .. string.format("%.3f", local_prediction)
        message = message .. " \ttruth: " .. string.format("%.3f", minibatch_dates[i])
        message = message .. " \tdiff: " ..  string.format("%.3f", math.abs(local_prediction - minibatch_dates[i]))
        message = message .. " \ttotal error: " .. string.format("%.3f", err)
    end
    return message
end

function epoch_summary(epoch_index, epochs, err_sum, minibatch_size, starting_time)
    local fraction_done = epoch_index / epochs
    local estimated_remaining_time = get_remaining_time(starting_time, fraction_done)
    local message = "[" .. epoch_index .. "/" .. epochs .. "] "
    message = message .. "Error rate: " .. err_sum / minibatch_size
    message = message .. "\tRemaining time: " .. estimated_remaining_time
    return message .. "\n"
end


local log_threshold = 1
function set_log_level(log_level) 
    log_threshold = log_level
end

function log(log_level, message)
    if log_level >= log_threshold then
        print(message)
    end
end

function median(list)
    local temp = {}
    for key, value in pairs(list) do
        if type(value) == 'number' then
            table.insert(temp, value)
        end
    end
    table.sort(temp)
    if math.fmod(#temp, 2) == 0 then
        return (temp[#temp/2] + temp[(#temp / 2) + 1]) / 2
    else
        return temp[math.ceil(#temp / 2)]
    end
end

function load_minibatch(params, frame_size, minibatch, shuffled_data)
    log(3, "Load thread: starting loading for minibatch " .. minibatch.index)

    for intra_minibatch_index = 1, params.minibatch_size do
        local abs_index = minibatch.index + intra_minibatch_index - 1
        log(1, "trying to load image to memory: " .. shuffled_data.files[abs_index])
        local frame = image.load(shuffled_data.files[abs_index], params.channels, 'double')
        local film = shuffled_data.films[abs_index]
        minibatch.frames[intra_minibatch_index] = frame
        minibatch.dates[intra_minibatch_index] = film.normalized_date
        minibatch.bins[intra_minibatch_index] = film.bin
        minibatch.probability_vectors[intra_minibatch_index] = film.bin_vector
        log(1, "loaded frame with normalized date " .. film.normalized_date[1])
    end

    log(3, "Load thread: loading complete for minibatch " .. minibatch.index)
end


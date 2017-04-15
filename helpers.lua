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

function minibatch_summary(minibatch_index, number_of_minibatches, epoch_index, epochs, starting_time, err_sum)
    local fraction_done  = ((epoch_index - 1) * number_of_minibatches + (minibatch_index - 1)) / (epochs * number_of_minibatches)
    local estimated_remaining_time = get_remaining_time(starting_time, fraction_done)
    local err = err_sum / minibatch_index * 1000
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
        message = message .. " \terror: " .. string.format("%.3f", err)
    end
    return message
end

function epoch_summary(epoch_index, epochs, err_sum, minibatch_size, starting_time)
    local fraction_done = epoch_index / epochs
    local estimated_remaining_time = get_remaining_time(starting_time, fraction_done)
    local message = "\nEPOCH " .. epoch_index .. " OF " .. epochs .. " COMPLETED"
    message = message .. "\nError rate: " .. err_sum / minibatch_size
    message = message .. "\nRemaining time: " .. estimated_remaining_time
    message = message .. "\n"
    return message
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

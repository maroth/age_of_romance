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

function epoch_summary(minibatch_index, number_of_minibatches, epoch_index, epochs, starting_time, err_sum)
    local fraction_done  = ((epoch_index - 1) * number_of_minibatches + (minibatch_index - 1)) / (epochs * number_of_minibatches)
    local elapsed_seconds = os.time() - starting_time
    local estimated_total_seconds = elapsed_seconds / fraction_done
    local estimated_remaining_seconds = estimated_total_seconds - elapsed_seconds
    local estimated_remaining_hours = estimated_remaining_seconds / 60 / 60
    local err = err_sum / minibatch_index
    return("[" .. epoch_index .. "/" .. epochs .. "] " .. " minibatch " .. minibatch_index .. " of " .. number_of_minibatches .. ", error rate: " .. string.format("%.3f", err) .. " (" .. string.format("%.3f", fraction_done * 100) .. "%, remaining hours: " .. string.format("%.5f", estimated_remaining_hours) .. ")")
end

function minibatch_summary(minibatch_size, prediction, minibatch_dates, err)
    local message = ""
    for i = 1, minibatch_size do
        message = message .. "\nprediction: " .. string.format("%.3f", prediction[i][1])
        message = message .. " \ttruth: " .. string.format("%.3f", minibatch_dates[i])
        message = message .. " \tdiff: " ..  string.format("%.3f", math.abs(prediction[i][1] - minibatch_dates[i]))
        message = message .. " \terror: " .. string.format("%.3f", err)
    end
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

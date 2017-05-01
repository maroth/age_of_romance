require 'nn'
require 'image'

require 'date_logic'
require 'load_logic'
require 'helpers'

function test(neural_network, criterion, params, test_frame_dir) 
    set_log_level(params.log_level)
    log(3, "\n\nTesting data with files from " .. test_frame_dir)
    local films = load_films(test_frame_dir, params.max_frames_per_directory, params.number_of_bins)
    log(3, "Number of test films: " ..  #films)

    local sorted_films = {}
    for _, film in pairs(films) do
        table.insert(sorted_films, film)
    end
    table.sort(sorted_films, function(a, b) return a.normalized_date[1] < b.normalized_date[1] end)

    local correct_predictions = 0
    local total_predictions = 0

    for _, film in ipairs(sorted_films) do

        local frame_count = 0
        for frame_index, frame_dir in pairs(film.frames) do
            local frame = image.load(frame_dir, params.channels, 'double')
            local minibatch = torch.DoubleTensor(1, frame:size(1), frame:size(2), frame:size(3))
            if (params.use_cuda) then
                minibatch = minibatch:cuda()
                frame = frame:cuda()
            end
            minibatch[1] = frame
            log(1, "feeding frame " .. frame_dir .. " to test network")

            local prediction = neural_network:forward(minibatch)
            frame_count = frame_count + 1

            local _, index = torch.max(prediction, 2)
            if index[1][1] == film.bin then
                correct_predictions = correct_predictions + 1
            end
            total_predictions = total_predictions + 1

        end
    end

    log(8, "\ntotal accuracy: " .. correct_predictions / total_predictions)
end


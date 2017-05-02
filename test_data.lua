require 'nn'
require 'image'

require 'date_logic'
require 'load_logic'
require 'helpers'
require 'optim'

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

    local correct_predictions = {}
    for i = 1, params.number_of_bins do
        correct_predictions[i] = 0
    end

    local total_predictions = 0

    local starting_time = os.time()
    local fraction_done = 0

    for film_index, film in ipairs(sorted_films) do
        log(8, "testing on film " .. film.title)

        local frame_count = 0
        for frame_index, frame_dir in pairs(film.frames) do
            status, err = pcall(function()
                local frame = image.load(frame_dir, params.channels, 'double')
                local minibatch = torch.DoubleTensor(1, frame:size(1), frame:size(2), frame:size(3))
                minibatch[1] = frame
                if (params.use_cuda) then
                    minibatch = minibatch:cuda()
                end
                log(1, "feeding frame " .. frame_dir .. " to test network")

                local prediction = neural_network:forward(minibatch)
                --print("\n\n\nbin", film.bin)
                --print(prediction)

                local best_prediction = params.number_of_bins
                for i = 1, params.number_of_bins do
                    values, indexes = prediction:topk(i, true, true)
                    --print("topk ", i, values, indexes)
                    for j = 1, i do
                        if film.bin == indexes[1][j] then
                            correct_predictions[i] = correct_predictions[i] + 1
                            if best_prediction > i then
                                best_prediction = i
                            end
                        end
                    end
                end

                print(3, "correct bin in top " .. best_prediction .. " probabilities")

                frame_count = frame_count + 1
                total_predictions = total_predictions + 1

            end) 
            if not status then
                log(9, "ERROR: could not load frame " .. frame_index .. " from directory " .. frame_dir .. " -- " .. err)
            end

        end

        fraction_done = film_index / #films
        local remaining_time = get_remaining_time(starting_time, fraction_done)
        log(7, "remaining time: " .. remaining_time)
    end


    logger = optim.Logger("test.log")
    logger:setNames{''}
    logger:style{'+-'}
    logger:display(false)
    for i = 1, params.number_of_bins do
        local accuracy = correct_predictions[i] / total_predictions
        log(8, "\ntotal top " .. i .. " accuracy: " .. accuracy)
        logger:add{accuracy}
    end
    logger:plot()
end


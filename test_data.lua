require 'nn'
require 'image'

require 'date_logic'
require 'load_logic'
require 'helpers'

function test(neural_network, criterion, params, test_frame_dir) 
    set_log_level(params.log_level)
    log(3, "\n\nTesting data with files from " .. test_frame_dir)
    local films = load_films(test_frame_dir, params.max_frames_per_directory)
    log(3, "Number of test films: " ..  #films)

    local sum_mean_error = 0
    local sum_median_error = 0

    local film_logger = optim.Logger('results.log')
    film_logger:setNames{'Truth', 'Mean Prediction', 'Median Prediction'}
    film_logger:style{'+-', '+-', '+-', '+-'}

    for _, film in pairs(films) do
        local sum_prediction = 0
        local frame_count = 0
        local predictions = {}
        for frame_index, frame_dir in pairs(film.frames) do
            local frame = image.load(frame_dir, 3, 'double')
            local minibatch = torch.DoubleTensor(1, frame:size(1), frame:size(2), frame:size(3))
            if (params.use_cuda) then
                minibatch = minibatch:cuda()
                frame = frame:cuda()
            end
            minibatch[1] = frame
            log(1, "feeding frame " .. frame_dir .. " to test network")
            local prediction = neural_network:forward(minibatch)
            log(1, "prediction: " .. prediction[1])
            sum_prediction = sum_prediction + prediction[1]
            local err = math.abs((prediction[1] - film.normalized_date)[1])
            frame_count = frame_count + 1
            table.insert(predictions, prediction[1])
        end

        local mean_prediction = sum_prediction / frame_count
        local median_prediction = median(predictions)

        local denormalized_mean = denormalize_date(mean_prediction)
        local denormalized_median = denormalize_date(median_prediction)

        local film_mean_error = math.abs((mean_prediction - film.normalized_date)[1])
        local film_median_error = math.abs((median_prediction - film.normalized_date)[1])
        
        sum_mean_error = sum_mean_error + film_mean_error
        sum_median_error = sum_median_error + film_median_error

        film_logger:add{film.normalized_date[1], mean_prediction, median_prediction}

        log(8, "")
        log(8, film.title)
        log(8, "actual date: " .. film.date ..  "\tmean prediction: " .. denormalized_mean .. "\tmedian prediction: " .. denormalized_median)
        --film_logger:plot()
    end

    local mean_error = sum_mean_error / #films
    local median_error = sum_median_error / #films
    log(8, "mean of mean error on test set: " .. mean_error)
    log(8, "mean of median error on test set: " .. median_error)


    return mean_error, median_error
end


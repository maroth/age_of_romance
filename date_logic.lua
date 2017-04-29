zero_date = os.time({year = 1960, month = 1, day = 1})
one_date = os.time({year = 2010, month = 1, day = 1})

function parse_date(date_to_convert)
    local pattern = "(%d+)-(%d+)-(%d+)"
    local runyear, runmonth, runday = date_to_convert:match(pattern)
    local time_stamp = os.time({year = runyear, month = runmonth, day = runday})
    return time_stamp
end

function normalize_date(date_to_normalize)
    if date_to_normalize >= one_date then
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
    local denormalized_date = date_to_denormalize * (one_date - zero_date) + zero_date
    local date_table = os.date("*t", denormalized_date)
    local date_string = date_table.year .. "-" .. date_table.month .. "-" .. date_table.day
    return date_string
end

function get_bin(normalized_date, number_of_bins)
    local bin = math.floor(normalized_date / (1 / number_of_bins)) + 1
    return bin
end

function create_probability_vector(normalized_date, number_of_bins)
    local position = get_bin(normalized_date, number_of_bins)
    local vector = torch.zeros(number_of_bins)
    vector[position] = 1
    return vector
end

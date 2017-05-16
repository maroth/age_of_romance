import os

frame_dir = "frames_211x176"

def create_distributed():
    os.system("mkdir train")
    os.system("mkdir test")
    os.system("mkdir validate")
    for _, dirs, _ in os.walk(frame_dir):
        for subdir in dirs:
            mkdir_command = "mkdir validate/" + subdir
            os.system(mkdir_command)
            mkdir_command = "mkdir train/" + subdir
            os.system(mkdir_command)
            mkdir_command = "mkdir test/" + subdir
            os.system(mkdir_command)

            copy_command = "cp " + frame_dir + "/" + subdir + "/info.json " + "validate/" + subdir
            os.system(copy_command)
            copy_command = "cp " + frame_dir + "/" + subdir + "/info.json " + "train/" + subdir
            os.system(copy_command)
            copy_command = "cp " + frame_dir + "/" + subdir + "/info.json " + "test/" + subdir
            os.system(copy_command)

            for x in range(2, 401, 5):
                copy_command = "cp " + frame_dir + "/" + subdir + "/" + str(x) + ".png " + "validate/" + subdir
                os.system(copy_command)

            for x in range(4, 401, 5):
                copy_command = "cp " + frame_dir + "/" + subdir + "/" + str(x) + ".png " + "test/" + subdir
                os.system(copy_command)

            for x in range(1, 401, 5):
                copy_command = "cp " + frame_dir + "/" + subdir + "/" + str(x) + ".png " + "train/" + subdir
                os.system(copy_command)
            for x in range(3, 401, 5):
                copy_command = "cp " + frame_dir + "/" + subdir + "/" + str(x) + ".png " + "train/" + subdir
                os.system(copy_command)
            for x in range(5, 401, 5):
                copy_command = "cp " + frame_dir + "/" + subdir + "/" + str(x) + ".png " + "train/" + subdir
                os.system(copy_command)


def create_continuous():
    os.system("mkdir train")
    os.system("mkdir test")
    os.system("mkdir validate")
    for _, dirs, _ in os.walk(frame_dir):
        for subdir in dirs:
            mkdir_command = "mkdir validate/" + subdir
            os.system(mkdir_command)
            mkdir_command = "mkdir train/" + subdir
            os.system(mkdir_command)
            mkdir_command = "mkdir test/" + subdir
            os.system(mkdir_command)

            copy_command = "cp " + frame_dir + "/" + subdir + "/info.json " + "validate/" + subdir
            os.system(copy_command)
            copy_command = "cp " + frame_dir + "/" + subdir + "/info.json " + "train/" + subdir
            os.system(copy_command)
            copy_command = "cp " + frame_dir + "/" + subdir + "/info.json " + "test/" + subdir
            os.system(copy_command)

            for x in range(81, 161, 1):
                copy_command = "cp " + frame_dir + "/" + subdir + "/" + str(x) + ".png " + "validate/" + subdir
                os.system(copy_command)

            for x in range(241, 321, 1):
                copy_command = "cp " + frame_dir + "/" + subdir + "/" + str(x) + ".png " + "test/" + subdir
                os.system(copy_command)

            for x in range(1, 81, 1):
                copy_command = "cp " + frame_dir + "/" + subdir + "/" + str(x) + ".png " + "train/" + subdir
                os.system(copy_command)

            for x in range(161, 241, 1):
                copy_command = "cp " + frame_dir + "/" + subdir + "/" + str(x) + ".png " + "train/" + subdir
                os.system(copy_command)

            for x in range(321, 401, 1):
                copy_command = "cp " + frame_dir + "/" + subdir + "/" + str(x) + ".png " + "train/" + subdir
                os.system(copy_command)



def create_seperate():
    os.system("mkdir train")
    os.system("mkdir test")
    os.system("mkdir validate")
    for year in range(1960, 2010, 1):
        year_films = []
        for subdirs, dirs, files in os.walk(frame_dir):
            for subdir in dirs:
                if subdir.startswith(str(year)):
                    year_films.append(subdir)
        year_films = sorted(year_films, key=lambda x: (int(x.replace('_', '-').split('-')[1]), int(x.replace('_', '-').split('-')[2])))
        copy_command = "cp -r " + frame_dir + "/" + year_films[0] + " train/"
        os.system(copy_command)
        copy_command = "cp -r " + frame_dir + "/" + year_films[1] + " validate/"
        os.system(copy_command)
        copy_command = "cp -r " + frame_dir + "/" + year_films[2] + " train/"
        os.system(copy_command)
        copy_command = "cp -r " + frame_dir + "/" + year_films[3] + " test/"
        os.system(copy_command)
        copy_command = "cp -r " + frame_dir + "/" + year_films[4] + " train/"
        os.system(copy_command)


#create_distributed()
create_continuous()
#create_seperate()
        

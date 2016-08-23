#!/usr/bin/python

datafile = "u.data"
moviefile = "u.item"
userfile = "u.user"
outputfile = "datafile.txt"
outputdata = []

#open all the required files
with open(datafile, 'r') as df, open(moviefile, 'r') as mf, open(userfile, 'r') as uf, open(outputfile, 'a') as of:
    datalines = df.readlines()
    movielines = mf.readlines()
    userlines = uf.readlines()
    
#get movieid and userid to search for them in moviefile and userfile
    for dataline in datalines:
        content = dataline.split()  # [0] = userid, [1] = movieid, [2] = rating, [3] = timestamp

#datalist is for storing all required data in single list
        datalist = []
        datalist.append(content[0])
        datalist.append(content[1])
        datalist.append(content[2])
        datalist.append(content[3])

#extracting the movie genres based on movieid
        for movieline in movielines:
            movieline = movieline.strip()   # to remove \n from last of the string
            moviecontent = movieline.split('|')
            if content[1] == moviecontent[0]:
                for i in range(6,24):
                    datalist.append(moviecontent[i])    # see how dates can also be entered
                break

#extracting the zip-code for user
        for userline in userlines:
            userline = userline.strip()
            usercontent = userline.split('|')
            if content[0] == usercontent[0]:
                datalist.append(usercontent[4])
                break

#converting the datalist to a comma separated string to be written in final output file
        datastring = ""
        datastring = datastring + datalist[0]
        for index,item in enumerate(datalist):
            if index != 0:
                datastring = datastring + ","
                datastring = datastring + item

        outputdata.append(datastring)

#writing the final output in the final file
    for output in outputdata:
        of.write("%s \n" % output)


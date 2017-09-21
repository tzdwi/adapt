# adapt: ARCTIC Data for PreMAP Term (I tried)
An easily-readable, modular pipeline to reduce some data from the ARCTIC camera on APO.

The data reduction portion of this does almost exactly what acronym (Weisenburger et al. 2017, https://github.com/kweis/acronym) does, with the exception of performing an overscan subtraction before trimming. I wrote this to be slightly more explicit (read: slower) so that a PreMAP student can run any portion of the reduction and see what it does. If you want to reduce ARCTIC data and you are not my PreMAP student, PLEASE use Kolby's code!

Hell, even if you are my PreMAP student, try Kolby's code too!

If you’re feeling lazy, you can run the entire pipeline by doing
python /path/to/adapt.py datadir caldir reddir
where datadir is the path to all of your data, and caldir and reddir are where the reduced master calibrations and science images go (respectively). If caldir and reddir don’t yet exist, adapt will make them. If datadir doesn’t exist, idk ur kinda SOL lol where even is your data bruh.

If you don’t give adapt.py those directories (or give them the wrong number of directories, whoops), then data are assumed to be in your current directory, and the reduced data will go into the same directory. No worries, adapt won’t mess with your original files, so you can always delete the reduced data and try again if something goes wrong.

If you want more functionality, you can import run_pipeline_run from adapt.py, which has a few more options — namely the order of the polynomial used to fit the overscan region of the CCD images, and the prefix that goes in front of your reduced science images. Even more customization is available by digging into the pipeline functions that create master cals, or fetch the appropriate cals, or reduce the science images etc. 

If this is your first time reducing data (which, if you’re using this code instead of acronym, is probably true), then I encourage you to at least look at what run_pipeline_run does, and try doing the individual steps for one science image, just to see what the intermediate processes all look like!

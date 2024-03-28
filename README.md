# How to Use
## Image Folder
Create a folder with your images.

Include the type of the image (e.g. Anime Character) in its filename if you want the slide to be labeled.\
This will also generate a list of the types of images that are in the quiz on the Rules slide.\
Here is a list of the currently accepted types of images:\
Location\
Duo\
Weapon or Ability\
Pokemon\
Famous Person\
Cartoon Character\
Character\
Scene\
The Archive\
To make the type of image "Anime", also include "Anime" in the picture title.\
This is optional. The slides will just have a slide number without the image type in the top left if you don't provide one.\
Also, capitalization doesn't matter, but the above names are what will appear on the slides (with "Anime " appended to the beginning if it's anime).

If you have a title slide picture, include "Title" in the picture's file name.

## Generating the Powerpoint
Either drag the images folder onto quizMaker.exe or run quizMaker.exe with the images folder in the same directory and call it "images".\
A console will appear that prints the name of each image when it gets to it so that you can see the progress.\
The powerpoint will be called "quiz.pptx" and it will be outputted in the same directory as the images folder.\
Note that the rules slide may go off the page. This can be fixed by just opening the presentation and typing something into the textbox (you can then remove whatever you typed).\
This could kind of be done in code, but the way I see to do it has worse results than what the above method provides. Better results would be a bit overly annoying to code unless somebody's already done it somewhere online.

## The result
The default quiz title is just "Hal Quiz". You can just change this to whatever.\
If you don't have a title picture, then the title text is black by default. If you do have a title picture, it is white.\
If you have a title picture, it is fitted to the size of the slide and is put behind the white title.

The Rules slide is titled "Rules" with the first bullet point being "Categories (Indicated on each page):".\
If you didn't label any images, then it just lists "secret". If you did, then it lists every type of image in the quiz separated into "Anime" and "Non-Anime" if applicable.

Each slide after this has the image type of the image being guessed in the top left corner, unless no image type was provided.\
Each slide also has their slide number starting from the first image slide in the top left corner. It is under the image type if the image type is there.\
The images hints are set up the same way as we do it normally with 3 slides for hints where subsequent hints are copied over to each slide, plus a fourth slide with the entire image.

# How to Build
To build the .exe file, run `pyinstaller quizMaker.spec`.\
Install pyinstaller with pip if you don't have it.

## Don't rebuild the .spec file. 
I had to fix it based off this solution: \
https://stackoverflow.com/questions/68099452/is-there-a-way-to-use-pptx-python-with-python-verion-3-7-and-make-exe-with-pyins 

Add these lines at the top: 
```import sys # added line
from os import path # added line 
site_packages = next(p for p in sys.path if 'site-packages' in p) # added line ```

Add this to the `datas` array in Analysis: 
`(path.join(site_packages,"pptx","templates"), "pptx/templates") `

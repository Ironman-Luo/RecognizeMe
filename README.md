# RecognizeMe

In this project we are implementing this recognizeme app.
If you click login:
  we will capture two photos from you.
  first photo is captured once you click login.
  second photo is captured after one second you click login.
Then we will match yout face to our existing database, and once a match has been formed, you will return to welcome page.
If no face is detected or no face is matched, you are not welcomed.
if you click sign up, we will capture one photo from you and this photo will be added to our database.

#Instructions:
In order to get all packages setup, you need to run 

```
  pip install -r requirements.txt
```

# Note: Please Be Sure to let the camera capture you both open and close eyes
Normal version does not do the extra_credit work.
If you want to test for extra credit:
  set variable extra_credit in app.py submit() function to True.
If you are a real human, you can open your eye for half a second then close your eye for one second in order to login.
If the camera did not capture both your closed_eye picture and open_eye picture, it will not let you in.

(Do not use HeroKu App link for website, it is not set up yet)


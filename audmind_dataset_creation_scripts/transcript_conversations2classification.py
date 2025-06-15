import pandas as pd
import openai
import os
import json
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    api_key = "your-api-key")


examples_task1 = [
    {'Text': "As I kid, I had been pretty anxious and nervous, but they were negligible. I only got anxious when there was something happening, like giving presentation etc. They did not tamper with my everyday life and I was not concerned with them. But recently, I felt like my anxiety has developed into a new level. I got so damn anxious for every day stuff." , 'Question': "Does the patient suffer from stress?"},	
    {'Text': "Instead he followed me to the store in his car. He would even drive slow enough to have the car next to me at all times. During one of our fights he took a permanent marker and slashed lines all over me while telling me 'how does that feel, you slut?' He walked away and I just fell to the floor in tears. He came back and poured his Gatorade all over me telling me to shut up and stop crying so loud" , 'Question': "Does the patient suffer from stress?" },
    {'Text': "Who even cares anymore I’ve just learned to accept my thought and I realise things aren’t gonna get better.I have the same cycle every day just play video games mourn for death wish I had weed and listen to sad songs till the morning life has just become such a boring shit show. Even talking to random people has me fucked up cause i fuck things up tell me guys what’s the point of even living anymore", 'Question': "Does the patient suffer from depression?" },
    {'Text': "The one and only thing I want to and have a will to do is to listen to the music all day and space out in it. Everything else is boring, draining and dull. And I receive no mental pleasure from doing anything else.", 'Question': "Does the patient suffer from depression?" },
    {'Text': "I had a week off over Christmas. My family is all overseas so I was not busy or rushed. I took things easy and slow and didn't do much. It really alleviated the pressures of life and did help my depression a bit. Now that I'm back at work I feel incredibly depressed - way more than before. I had a taste of feeling relaxed but crashed to the ground. I need time off to just sort out what I want to do in life, but at work during the week it's all rush rush rush and stress and chores and anxiety. At the end of the day I am just exhausted and do need to sleep or rest. I know what helps - having time off to just figure stuff out. But it's just so unobtainable.", 'Question': "Does the patient suffer from loneliness?"}
]

examples_task2 = [
    {'Text': "Effexor day 1 is going badly Good news! So the nausea has died away. Bad news... Someone knocked at the door and was acting weird so I went into full panic mode. I'm worried that the owners of the abused dog I'm fostering will come round or send over goons which is why I went to the doc for something. I'm not too sure he was right to put me on a/d instead of something prn. I ran around the house making sure all windows are locked. I set up an old phone in the front door window recording video. I've got a hammer by the back door and exit plans for all contingencies. But I can't relax. My heart is going a mile a minute. My stomach is in even more knots than before and my breathing is more shallow. Bleugh! Update - and I just got a threatening message from the not so nice lady who caused criminal damage to my door. Just when the anxiety starts going down they spike it back up.", 'Question': "What mental disorder symptoms does this patient show?"},
    {'Text': "I really fucking hate everything and everybody the same. This life is not worth living, love ain't like the movies, yes, we're fucked!" , 'Question': "What mental disorder does this patient show?"},
    {'Text': "Don't ever try to change someone. If you have to change them, then they're obviously not right for you anyway. Why waste…", 'Question': "What mental disorder does this patient show?" },
    {'Text': "Guided Meditation Disclaimer: I am in no way saying meditation should be the sole method of treating bipolar disorder. I wouldn't be stable enough to meditate without my meds. Just sharing an experience. I attended my first guided meditation session today and found it really helpful. My mind has been racing for days. Just random thoughts pushing out other random thoughts. 25 minutes in a dimly-lit room listening to the soothing voice of someone guiding images was nice. My mind wandered as she read, but she instructed me to acknowledge the thoughts and gently push them away. PTSD tells me not to close my eyes in the room with a stranger; bipolar disorder tells me I can't sit still (mentally or physically) long enough to get through a guided meditation. But get through it I did, and it was very helpful. It was conducted in a little shop in our downtown area and only cost $5. If you're interested in meditation, it might be something to check out in your area. Just my experience." ,'Question': "What mental disorder symptoms does the patient show?"}    
]

examples_task3 = [
    {'Text': "The coronavirus epidemic is frightening me and causing intense worry to me and my family" , 'Question': "What is the cause of stress?"},
    {'Text': "The most heartbreaking thought I have is my mom's thoughts when she knows her kid wants to die. She doesnt know how to help. I dont know how to help either tho... Some fuckin weed would be a good start tho lol", 'Question': "What cause of depression does this show?"},
    {'Text': "I dont have any time to study which i despretely need to do since i'm not doing to good in school at the moment.", 'Question': "What is the stress cause here?"},
    {'Text': "In bad news my sister has tested positive for ovarian cancer..my family is devastated." , 'Question': "What is the stress cause here?"},
    {'Text': "yeh, we talked for like 10 minutes but had to go cuz i had to finish work at starbucks.", 'Question': "What is the stress cause here?"}
]

examples_task4 = [
    {'Text': "Im afraid. The suicide scale is true. Its gets progressively worse and you dont even realize it. I was at 1 contemplating death and that was it. Now as i live i keep moving farther into the scale. Not existing is all i long for. Its all i think about. Its all i desire. Ive written 2 goodbye notes. I constantly open up the lid of pills and acting as if ill swallow them. Yesterday and today i have been making nooses with a thin belt and putting it around my neck. I squeeze so hard to simulate what its like. When i do i can feel the pressure in the top sides of my head. I can feel the rush and fear. I stopped giving a shit about everything and my attention is mainly death. Someone please comfort me. I am afraid.", 'Question': "Does the patient show risk of thwarted belongingness?"},
    {'Text': "I want to feel like I belong here. I wish there were people whom I could connect with. Unfortunately, I don't know where life is taking me right now. I'm too tired of the everyday grind. Life has become a constant struggle. I wish this agony would end. I've been hoping for a long time. I thought I had people by my side. There are people who care but I don't know why I'm not able to connect with them. I just feel empty. I've endured for a long time. I just wish this would end some day", 'Question': "Does the patient show risk of perceived burdensomeness?"},
    {'Text': "I feel like everyone just doesn't appreciate me. And I only feel this because of school. My teacher told me today that if I didn't do anything that would be more helpful than me actually trying to do something. I'm only 10 and I feel like no one my age should be feeling like this. I know you people are much older and have to deal with way worse problems but when your a goddamn 5th grader life shouldn't make you feel like your worthless. My entire day was filled with unfairness. Teachers have their favorites and if you aren't one of them you'll be treated like shit.", 'Question': "Does the social wellness dimension exist here?"},
    {'Text': "I cannot believe how fucking dumb I am. I fucking ruined one of the best things that happened to me. I want to die . I want to die. I need to kill myself soon. Please god don‚Äôt let me wake up tomorrow", 'Question': "Does the emotional wellness dimension exist here?"},
    {'Text': "Everything is getting tougher for me , most of my problems stem from my at home life and with my community. I do go to college but I can't apply for dorms till September and if I get them I don't know if fafsa would cover it or not. I really do believe I might not survive being home any longer then I already have , honestly this pains me to admit, I love my family I really do and my friends but I am breaking from the inside out , I need a out. My question : is there a way to get shelter or government form of shelter I could apply for ? ( I am 22 and don't have any savings, current part time job , I only make $130-180 max) I ll try my best to follow the advice , this will be a scary step to take if taken , only my eldest sibling knows of my depression.", 'Question': "Does the vocational wellness dimension exist here?"},
    {'Text': "Im feeling so good and wouldnt even think about suicide and then hours go by and I donot remember what really happens but I feel like i cannot take it anymore. I dont really know how I got from point a to b but i did and i feel so awful and suicide becomes a realistic feeling and temptation", 'Question': "Does the emotional wellness dimension exist here?"},
    {'Text': "my life is worth nothing, yesterday my roommate raped me and now i'm scared of him, me and my bf broke up, it's just time to end this, if i end it now i won't feel like i have anything that i might as well finish, i have set the time, 6:30 tonight. is there any better way than a gun", 'Question': "Does the physical wellness dimension exist here?"}  
]

# Format few-shot examples
formatted_examples_t1 = "\n\n".join([f"Text: {ex['Text']}\nQuestion: {ex['Question']}" for ex in examples_task1])
formatted_examples_t2 = "\n\n".join([f"Text: {ex['Text']}\nQuestion: {ex['Question']}" for ex in examples_task2])
formatted_examples_t3 = "\n\n".join([f"Text: {ex['Text']}\nQuestion: {ex['Question']}" for ex in examples_task3])
formatted_examples_t4 = "\n\n".join([f"Text: {ex['Text']}\nQuestion: {ex['Question']}" for ex in examples_task4])

#---------------------------------------------------------------------------------------------

def classify_text_task1(text):
    prompt = f"""You are given a short piece of text spoken by a patient or a doctor in a clinical interview. You are also given examples of texts along with the type of questions that can be asked based on them.

Your task is:
Decide whether the given text is suitable enough for forming a question like the ones shown below. If yes, return 1. If not, return 0. Do not generate the question. Only classify.

Examples:
{formatted_examples_t1}

Now classify this:

Text: "{text}"
Label:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for mental health classification."},
            {"role": "user", "content": prompt}
        ]
        # , api_key="your-api-key"
    )

    return response.choices[0].message.content.strip()



def classify_text_task2(text):
    prompt = f"""You are given a short piece of text spoken by a patient or a doctor in a clinical interview. You are also given examples of texts along with the type of questions that can be asked based on them.

Your task is:
Decide whether the given text is suitable enough for forming a question like the ones shown below. If yes, return 1. If not, return 0. Do not generate the question. Only classify.

Examples:
{formatted_examples_t2}

Now classify this:

Text: "{text}"
Label:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for mental health classification."},
            {"role": "user", "content": prompt}
        ]
        # , api_key="your-api-key"
    )

    return response.choices[0].message.content.strip()



def classify_text_task3(text):
    prompt = f"""You are given a short piece of text spoken by a patient or a doctor in a clinical interview. You are also given examples of texts along with the type of questions that can be asked based on them.

Your task is:
Decide whether the given text is suitable enough for forming a question like the ones shown below. If yes, return 1. If not, return 0. Do not generate the question. Only classify.

Examples:
{formatted_examples_t3}

Now classify this:

Text: "{text}"
Label:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for mental health classification."},
            {"role": "user", "content": prompt}
        ]
        # , api_key="your-api-key"
    )

    return response.choices[0].message.content.strip()



def classify_text_task4(text):
    prompt = f"""You are given a short piece of text spoken by a patient or a doctor in a clinical interview. You are also given examples of texts along with the type of questions that can be asked based on them.

Your task is:
Decide whether the given text is suitable enough for forming a question like the ones shown below. If yes, return 1. If not, return 0. Do not generate the question. Only classify.

Examples:
{formatted_examples_t4}

Now classify this:

Text: "{text}"
Label:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for mental health classification."},
            {"role": "user", "content": prompt}
        ]
        # , api_key="your-api-key"
    )

    return response.choices[0].message.content.strip()


#------------------------------------------------------------------------------------

# Directories
input_dir = "yet_to_be_classified"
output_dir = "Transcriptions_classified_part2"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all CSV files
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_dir, filename)
        print(f"Processing: {file_path}")

        df = pd.read_csv(file_path)

        tqdm.pandas()
        df['task1'] = df['text'].progress_apply(classify_text_task1)
        df['task2'] = df['text'].progress_apply(classify_text_task2)
        df['task3'] = df['text'].progress_apply(classify_text_task3)
        df['task4'] = df['text'].progress_apply(classify_text_task4)
        output_filename = filename.replace(".csv", "_classified.csv")
        output_path = os.path.join(output_dir, output_filename)
        df.to_csv(output_path, index=False)

        print(f"Saved to: {output_path}")


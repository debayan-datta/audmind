import os
import pandas as pd
from tqdm import tqdm

from openai import OpenAI

client = OpenAI(
    api_key = "your-api-key")

# ---------------------------------- TASK 1 ----------------------------------------------------
def get_quest_ans_reason_task1(text):
    prompt = f"""
You are a mental health analysis assistant. You will be given a text written by a person, and your task is to:

1. Choose the most appropriate question from the following list that best matches the concerns expressed in the text:
- Does the patient suffer from stress?
- Does the patient suffer from depression?
- Does the patient suffer from loneliness?

2. Then, generate a short answer and the reasoning in the format:
"Question: ...   
Answer: ...   
Reasoning: ..."  

Refer to the following examples:

---
Example 1:
Text: "As I kid, I had been pretty anxious and nervous, but they were negligible. I only got anxious when there was something happening, like giving presentation etc. They did not tamper with my everyday life and I was not concerned with them. But recently, I felt like my anxiety has developed into a new level. I got so damn anxious for every day stuff."  

Model output: 
Question: Does the patient suffer from stress? 
Answer: Yes, the patient suffers from stress. 
Reasoning: The patient openly discusses their history of anxiety and how it used to only affect them during specific events. However, they now express that their anxiety has escalated to a new level and is impacting their day-to-day life. This indicates that the patient is experiencing increased stress and anxiety symptoms, suggesting they are suffering from ongoing stress.	
---
Example 2:
Text: "Instead he followed me to the store in his car. He would even drive slow enough to have the car next to me at all times. During one of our fights he took a permanent marker and slashed lines all over me while telling me 'how does that feel, you slut?' He walked away and I just fell to the floor in tears. He came back and poured his Gatorade all over me telling me to shut up and stop crying so loud" 

Model output:
Question: Does the patient suffer from stress? 
Answer: Yes, the patient suffers from stress. 
Reasoning: The language used here is indicative of a traumatic and abusive relationship. The patient describes being followed and harassed by their partner, as well as being physically and emotionally abused. The extreme level of distress and fear expressed, such as falling to the floor in tears, suggests that the patient is experiencing severe stress. The traumatic events described here would likely have a significant impact on the patient's mental and emotional well-being.
---
Example 3:
Text: "Who even cares anymore I’ve just learned to accept my thought and I realise things aren’t gonna get better.I have the same cycle every day just play video games mourn for death wish I had weed and listen to sad songs till the morning life has just become such a boring shit show. Even talking to random people has me fucked up cause i fuck things up tell me guys what’s the point of even living anymore" 

Model output:
Question: Does the patient suffer from depression? 
Answer: Yes, the patient suffers from depression. 
Reasoning: The patient suggests a deep sense of apathy, hopelessness, and a lack of motivation or enjoyment in life. The patient describes engaging in activities such as playing video games, mourning, and listening to sad songs as a way to cope, but even these activities seem to bring little joy or fulfillment. The mention of wanting weed as a means of escape suggests a desire to numb or escape from emotional pain. The question about the point of living anymore highlights a significant level of despair and questioning of the meaning or purpose of life. These emotional indicators align with symptoms commonly associated with depression.
---
Example 4:
Text: "The one and only thing I want to and have a will to do is to listen to the music all day and space out in it. Everything else is boring, draining and dull. And I receive no mental pleasure from doing anything else." Question: Does the patient suffer from depression?

Model output:
Question: Does the patient suffer from depression? 
Answer: Yes, the patient suffers from depression. 
Reasoning: The patient expresses a lack of interest, enjoyment, and pleasure in activities other than listening to music. The patient describes everything else as boring, draining, and dull, indicating a lack of motivation or fulfillment in their daily life. This aligns with symptoms commonly associated with depression, such as anhedonia (loss of interest or pleasure in activities) and lack of motivation. The emphasis on wanting to listen to music all day and ""space out"" suggests a desire to escape or disconnect from their current emotional state. Overall, it conveys a sense of emotional distress and lack of fulfillment that is consistent with depression.
---
Example 5:
Text: "I had a week off over Christmas. My family is all overseas so I was not busy or rushed. I took things easy and slow and didn't do much. It really alleviated the pressures of life and did help my depression a bit. Now that I'm back at work I feel incredibly depressed - way more than before. I had a taste of feeling relaxed but crashed to the ground. I need time off to just sort out what I want to do in life, but at work during the week it's all rush rush rush and stress and chores and anxiety. At the end of the day I am just exhausted and do need to sleep or rest. I know what helps - having time off to just figure stuff out. But it's just so unobtainable." 

Model output:
Question: Does the patient suffer from loneliness?
Answer: Yes, the patient suffers from loneliness. 
Reasoning: The patient mentions feeling depressed before and after the week off over Christmas. The patient describes the time off as alleviating the pressures of life and helping with their depression, suggesting a longing for more downtime and relaxation. They mention feeling incredibly depressed, rushed, stressed, and anxious at work, emphasizing the negative emotions they experience in their daily life. They express a need for time off to figure things out and state that it feels unobtainable, indicating a sense of longing and frustration. The overall tone here suggests a deep sense of loneliness and a desire for a change in their current circumstances.
---
Now, based on the following input, follow the same format:

Text: “{text}”

Your response:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    # return response['choices'][0]['message']['content'].strip()
    # print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()

# ---------------------------------- TASK 2 ---------------------------------------------------
def get_quest_ans_reason_task2(text):
    prompt = f"""
You are a mental health analysis assistant. You will be given a text written by a person, and your task is to:

1. Choose the most appropriate question from the following list that best matches the concerns expressed in the text:
- What mental disorder does this person show?
- What mental disorder symptoms does the patient show?

2. Then, generate a short answer and the reasoning in the format:
"Question: ...   
Answer: ...   
Reasoning: ..."  

Refer to the following examples:

---
Example 1:
Text: "Effexor day 1 is going badly Good news! So the nausea has died away. Bad news... Someone knocked at the door and was acting weird so I went into full panic mode. I'm worried that the owners of the abused dog I'm fostering will come round or send over goons which is why I went to the doc for something. I'm not too sure he was right to put me on a/d instead of something prn. I ran around the house making sure all windows are locked. I set up an old phone in the front door window recording video. I've got a hammer by the back door and exit plans for all contingencies. But I can't relax. My heart is going a mile a minute. My stomach is in even more knots than before and my breathing is more shallow. Bleugh! Update - and I just got a threatening message from the not so nice lady who caused criminal damage to my door. Just when the anxiety starts going down they spike it back up."

Model output: 
Question: What mental disorder symptoms does this patient show?
Answer: This shows mental disorder symptoms related to anxiety. 
Reasoning: The description includes multiple descriptions and symptoms commonly associated with anxiety, such as experiencing panic mode, worrying about potential threats, engaging in safety behaviors (locking windows, setting up video recording, having an exit plan), and physical symptoms like rapid heart rate, stomach knots, and shallow breathing. The mention of receiving a threatening message further supports the presence of anxiety. The individual also expresses feeling unable to relax and experiencing spikes in anxiety due to external events. 
---
Example 2:
Text: "I really fucking hate everything and everybody the same. This life is not worth living, love ain't like the movies, yes, we're fucked!".

Model output:
Question: What mental disorder does this person show?
Answer: This shows suicide or self-harm tendency. 
Reasoning: The use of phrases like ""I hate everything and everybody"" and ""This life is not worth living"" indicate a strong sense of hopelessness and despair. The mention of love not being like the movies and the phrase ""we're fucked"" further suggests a negative perspective on life and relationships. The use of profanity and the intensity of the emotions expressed suggest a high level of distress and potential risk for self-harm or suicide.
---
Example 3:
Text: "Don't ever try to change someone. If you have to change them, then they're obviously not right for you anyway. Why waste…".

Model output:
Question: What mental disorder does this patient show?
Answer: This shows depression. 
Reasoning: There are a few elements here that suggest the label of depression
---
Example 4:
Text: "I am in no way saying meditation should be the sole method of treating bipolar disorder. I wouldn't be stable enough to meditate without my meds. Just sharing an experience. I attended my first guided meditation session today and found it really helpful. My mind has been racing for days. Just random thoughts pushing out other random thoughts. 25 minutes in a dimly-lit room listening to the soothing voice of someone guiding images was nice. My mind wandered as she read, but she instructed me to acknowledge the thoughts and gently push them away. PTSD tells me not to close my eyes in the room with a stranger; bipolar disorder tells me I can't sit still (mentally or physically) long enough to get through a guided meditation. But get through it I did, and it was very helpful. It was conducted in a little shop in our downtown area and only cost $5. If you're interested in meditation, it might be something to check out in your area. Just my experience.".

Model output:
Question: What mental disorder symptoms does the patient show?
Answer: This shows mental disorder symptoms related to bipolar disorder. 
Reasoning: The description mentions having bipolar disorder and describes symptoms that are commonly associated with the condition, such as racing thoughts and difficulty sitting still mentally or physically. The mention of PTSD suggests the possibility of comorbidity or previous trauma, which can further impact the individual's ability to engage in activities like guided meditation.
---
Now, based on the following input, follow the same format:

Text: “{text}”

Your response:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    # return response['choices'][0]['message']['content'].strip()
    # print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()



# ----------------------------------- TASK 3 ------------------------------------------------
def get_quest_ans_reason_task3(text):
    prompt = f"""
You are a mental health analysis assistant. You will be given a text written by a person, and your task is to:

1. Choose the most appropriate question from the following list that best matches the concerns expressed in the text:
- What is the stress cause here?
- What cause of depression does this show?

2. Then, generate a short answer and the reasoning in the format:
"Question: ...   
Answer: ...   
Reasoning: ..."  

Refer to the following examples:

---
Example 1:
Text: "The coronavirus epidemic is frightening me and causing intense worry to me and my family"

Model output:
Question: What is the cause of stress?
Answer: This patient shows the stress cause related to financial problem. 
Reasoning: The patient explicitly mentions the coronavirus epidemic causing intense worry to the patient and their family. While the coronavirus epidemic is a global health crisis and can cause stress in various ways, one aspect that stands out here is the financial impact. The pandemic has led to widespread job losses, business closures, and economic uncertainties, which can directly contribute to financial problems and anxieties. Therefore, the most appropriate label is Financial Problem as the cause of stress.
---
Example 2:
Text: "The most heartbreaking thought I have is my mom's thoughts when she knows her kid wants to die. She doesnt know how to help. I dont know how to help either tho... Some fuckin weed would be a good start tho lol"

Model output:
Question: What cause of depression does this show?
Answer: This patient shows causes of depression related to medication. 
Reasoning: The patient mentions the her feelings of wanting to die and their concern for their mother's thoughts. It also mentions that they don't know how to help themselves. The mention of weed as a possible solution suggests that the patient may be seeking some form of self-medication to cope with their mental state. Therefore, the assigned label is Medication.
---
Example 3:
Text: "I dont have any time to study which i despretely need to do since i'm not doing to good in school at the moment."

Model output:
Question: What is the stress cause here?
Answer: This patient shows the stress cause related to school. 
Reasoning: The patient explicitly mentions that the patient does not have any time to study, which is something they desperately need to do because they are not doing well in school. This indicates that the stress is directly related to their academic performance and the pressures and responsibilities associated with school. Therefore, the appropriate label is School.
---
Example 4:
Text: "In bad news my sister has tested positive for ovarian cancer..my family is devastated."

Model output:
Question: What is the stress cause here?
Answer: This patient shows the stress cause related to family issues. 
Reasoning: The patient mentions that the her sister has tested positive for ovarian cancer, which is a significant health issue that affects the entire family. This news is causing devastation within the family, indicating that the stress is primarily related to family issues. Although health issues could also be a relevant label, the mention of the family's devastation suggests that the focus of the stress is on the impact this diagnosis has on the family as a whole. Therefore, the appropriate label is Family Issues.
---
Example 5:
Text: "yeh, we talked for like 10 minutes but had to go cuz i had to finish work at starbucks."

Model output:
Question: What is the stress cause here?
Answer: This patient shows the stress cause related to everyday decision making. 
Reasoning: The patient mentions having to end a conversation because of needing to finish work at Starbucks. This suggests that the patient is experiencing stress due to having to make a decision between spending time talking and completing work. This falls under the category of everyday decision making as a stress cause.
---
Example 6:
Text: "I never knew my father he left before i was born. My mother told me he started a charity for boys without father's after he left that and his first name is all i know about him. Four years ago my mom and stepfather made a move overseas from my home country. The transition has been difficult i feel alienated and haven't made any friends for four years other than brief online relationships. Unfortunately the move also took me out of school past the age of 13 i haven't had an education. My mom is to busy to maintain homeschool and my stepdad doesn't care for me as he does my mom. I was born in another country and he already has two children of his own I guess this is why he doesn't pay attention to me. It's hard being alone or feeling that way I've always wanted to meet someone my own age I dream of it. Some days i wake up and I'll sneak out of the house to the park and hope to meet someone. I haven't yet. My 18th birthday is happening this year and that's a big number to me but I'm afraid nothing is going to change its like I'm stuck in a purgatory while everyone else gets to go to school and have friends get a job. I'm in a village with a handful of houses in a country i wasnt born in. I have dreams and I even turned my room into a little acomplishement zone to achieve mini goals so that one day I'll be able to accomplish those dreams. Sometimes things just seem unfair i feel like I'm chipping away at a granite wall with a plastic spoon. My parents dont listen for years i begged them for an activity or something to give me growth and interaction with other kids it never happened. What's the purpose for all this happening why have i been put in a time of obscurity for this long? Sorry for the rant has anyone had a similar experience of being alone like this. I'm sure alot of people who grew up homeschooled experienced what im feeling." 

Model output: 
Question: What cause of depression does this person show? 
Answer: This patient shows causes of depression related to relationship. 
Reasoning: The patient describes the his feelings of loneliness and isolation, as well as their desire for relationships and connections with others. They mention not having a father figure in their life, a move to a new country, lack of education, and feeling neglected by their stepfather. These factors contribute to a sense of alienation and difficulty forming relationships. It also mentions seeking out opportunities to meet people but being unsuccessful, further highlighting the impact of their lack of relationships on their mental well-being. Therefore, the assigned label for the cause of mental disorders in here is relationship. 
---
Now, based on the following input, follow the same format:

Text: “{text}”

Your response:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    # return response['choices'][0]['message']['content'].strip()
    # print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()



# -------------------------------- TASK 4 ----------------------------------------------------
def get_quest_ans_reason_task4(text):
    prompt = f"""
You are a mental health analysis assistant. You will be given a text written by a person, and your task is to:

1. Choose the most appropriate question from the following list that best matches the concerns expressed in the text:
- Does the patient show risk of thwarted belongingness?
- Does the patient show risk of perceived burdensomeness?
- Does the physical wellness dimension exist here?
- Does the social wellness dimension exist here?
- Does the vocational wellness dimension exist here?
- Does the emotional wellness dimension exist here?
- Does the intellectual wellness dimension exist here?

2. Then, generate a short answer and the reasoning in the format:
"Question: ...   
Answer: ...   
Reasoning: ..."  

Refer to the following examples:
---
Example 1:
Text: "Im afraid. The suicide scale is true. Its gets progressively worse and you dont even realize it. I was at 1 contemplating death and that was it. Now as i live i keep moving farther into the scale. Not existing is all i long for. Its all i think about. Its all i desire. Ive written 2 goodbye notes. I constantly open up the lid of pills and acting as if ill swallow them. Yesterday and today i have been making nooses with a thin belt and putting it around my neck. I squeeze so hard to simulate what its like. When i do i can feel the pressure in the top sides of my head. I can feel the rush and fear. I stopped giving a shit about everything and my attention is mainly death. Someone please comfort me. I am afraid."

Model output:
Question: Does the patient show risk of thwarted belongingness?
Answer: Yes, the patient shows this risk. 
Reasoning: 1. The individual expresses fear and talks about being on the suicide scale, indicating a state of psychological distress. 2. They mention contemplating death and constantly thinking about it, suggesting a preoccupation with self-harm and a lack of hope. 3. The individual discusses engaging in self-destructive behaviors, such as writing goodbye notes and simulating asphyxiation, which indicates a risk of self-harm or suicide. 4. They express a desire for comfort and support, indicating a need for connectedness and belongingness that is currently unmet.
---
Example 2:
Text: "I want to feel like I belong here. I wish there were people whom I could connect with. Unfortunately, I don't know where life is taking me right now. I'm too tired of the everyday grind. Life has become a constant struggle. I wish this agony would end. I've been hoping for a long time. I thought I had people by my side. There are people who care but I don't know why I'm not able to connect with them. I just feel empty. I've endured for a long time. I just wish this would end some day"

Model output:
Question: Does the patient show risk of perceived burdensomeness?
Answer: Yes, the patient shows this risk. 
Reasoning: 1. The patient mentions feeling like they don't belong and a desire to connect with people. These feelings of social isolation and a lack of connection can contribute to perceived burdensomeness. 2. The patient also mentions feeling tired of the everyday grind and life being a constant struggle. These feelings of exhaustion and difficulty in coping can contribute to a negative self-perception and a sense of burden. 3. The patient expresses a desire for the agony to end and a feeling of emptiness. This suggests a potential desire to escape from their current state, which can be indicative of perceived burdensomeness.
---
Example 3:
Text: "I feel like everyone just doesn't appreciate me. And I only feel this because of school. My teacher told me today that if I didn't do anything that would be more helpful than me actually trying to do something. I'm only 10 and I feel like no one my age should be feeling like this. I know you people are much older and have to deal with way worse problems but when your a goddamn 5th grader life shouldn't make you feel like your worthless. My entire day was filled with unfairness. Teachers have their favorites and if you aren't one of them you'll be treated like shit."

Model output:
Question: Does the social wellness dimension exist here?
Answer: Yes, this wellness dimension exists here. 
Reasoning: The patient acknowledges the interdependence between individuals in a school setting. The author feels unappreciated and expresses their frustration with unfair treatment by teachers and the existence of favoritism. The author also states that they feel worthless, highlighting their awareness of the interconnection between themselves and their peers.
---
Example 4:
Text: "I cannot believe how fucking dumb I am. I fucking ruined one of the best things that happened to me. I want to die . I want to die. I need to kill myself soon. Please god don‚Äôt let me wake up tomorrow"

Model output:
Question: Does the emotional wellness dimension exist here?
Answer: Yes, this wellness dimension exists here. 
Reasoning: The wellness dimension of emotional aspect exists in the patient. The individual is expressing feelings of self-blame, regret, and a desire for death. This demonstrates awareness of their own emotions and a lack of acceptance of their current state
---
Example 5:
Text: "Everything is getting tougher for me , most of my problems stem from my at home life and with my community. I do go to college but I can't apply for dorms till September and if I get them I don't know if fafsa would cover it or not. I really do believe I might not survive being home any longer then I already have , honestly this pains me to admit, I love my family I really do and my friends but I am breaking from the inside out , I need a out. My question : is there a way to get shelter or government form of shelter I could apply for ? ( I am 22 and don't have any savings, current part time job , I only make $130-180 max) I ll try my best to follow the advice , this will be a scary step to take if taken , only my eldest sibling knows of my depression."

Model output:
Question: Does the vocational wellness dimension exist here?
Answer: Yes, this wellness dimension exists here. 
Reasoning: The patient mentions the person's concerns about their living situation and the possibility of needing shelter or government assistance. This suggests that the vocational aspect of wellness is present, as it involves their work-related concerns and financial stability
---
Example 6:
Text: "Im feeling so good and wouldnt even think about suicide and then hours go by and I donot remember what really happens but I feel like i cannot take it anymore. I dont really know how I got from point a to b but i did and i feel so awful and suicide becomes a realistic feeling and temptation"

Model output: 
Question: Does the intellectual wellness dimension exist here?
Answer: No, this wellness dimension does not exist here. 
Reasoning: The patient primarily focuses on feelings of despair, not on any intellectual or cultural activities. There is no mention of engaging in any intellectual pursuits or participating in any cultural activities. Therefore, the patient does not indicate the presence of the intellectual aspect of psychological wellness.
---
Example 7:
Text : "my life is worth nothing, yesterday my roommate raped me and now i'm scared of him, me and my bf broke up, it's just time to end this, if i end it now i won't feel like i have anything that i might as well finish, i have set the time, 6:30 tonight. is there any better way than a gun".

Model output:
Question: Does the physical wellness dimension exist here?
Answer: Yes, this wellness dimension exists here. 
Reasoning: The wellness dimension of physical aspect does not exist here. It discusses traumatic experiences such as rape and a breakup, as well as contemplation of self-harm. These focus primarily on emotional and mental well-being, rather than growth, changes in the body, or the adoption of healthy habits.
---
Now, based on the following input, follow the same format:

Text: “{text}”

Your response:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    # return response['choices'][0]['message']['content'].strip()
    # print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()



# ------------------------------------- main function --------------------------------------------- # 
if __name__ == '__main__':
    
    input_dir = "Transcriptions_classified_all"
    output_dir = "Transcriptions_qar"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith("_classified.csv"):
            input_path = os.path.join(input_dir, filename)
            print(f"\nProcessing: {input_path}")
            
            df = pd.read_csv(input_path)
    
            # Task 1
            task1_results = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Task 1"):
                text = row['text']
                task1_results.append(get_quest_ans_reason_task1(text) if row['task1'] == 1 else None)
            df['QA for task1'] = task1_results
    
            # Task 2
            task2_results = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Task 2"):
                text = row['text']
                task2_results.append(get_quest_ans_reason_task2(text) if row['task2'] == 1 else None)
            df['QA for task2'] = task2_results
    
            # Task 3
            task3_results = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Task 3"):
                text = row['text']
                task3_results.append(get_quest_ans_reason_task3(text) if row['task3'] == 1 else None)
            df['QA for task3'] = task3_results
    
            # Task 4
            task4_results = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Task 4"):
                text = row['text']
                task4_results.append(get_quest_ans_reason_task4(text) if row['task4'] == 1 else None)
            df['QA for task4'] = task4_results
    
            # Save with _qar.csv suffix
            output_filename = filename.replace("_classified.csv", "_qar.csv")
            output_path = os.path.join(output_dir, output_filename)
            df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

            
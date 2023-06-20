# Playground (with a lot of stuff)
Another pure masterpiece from FPHam, Text WebUI extension to add clever multi-Notebook TAB to Chat mode

![image](https://github.com/FartyPants/Playground/assets/23346289/bb9899f0-b3dd-4b9a-8f4e-ba168c95484d)

## Features
- two independent Notebooks that are always present, regardless of the mode
- you can type in one while the other is generating.
- Quick instruct syntax
- Select and Insert - generate text in the middle of your text
- Perma Memory
- Summarization and ability to inject summary
- Paraphrasing
- Lora-Rama - to test various checkpoints when using LORA

## Quick instruct
The notebooks can enable Quick Instruct mode, where --- (three dashes) signal the line is an instruct (and will be hence formatted accordingly to User/bot prompt)

![image](https://github.com/FartyPants/Playground/assets/23346289/9320a2ec-9d17-45f7-936a-567cd0531447)

if you just type a question such as:
```
What is another word for happy?
```
the LLM may likely answer in another question:
```
What is another word for happy?
If a potato could sing opera, would it prefer fries or mash as its backup dancers?
```
but if you prepend three dashes
```
--- What is another word for happy?
```
this now means the line is an instruct and the LLM will respond according to it's fine tune
```
--- What is another word for happy?
Oh, there are loads of them. There's cheerful and joyous, euphoric. I suppose jubilant, that might be good, it's like happy in Spanish, isn't it?
--- You had been most helpful
Was I? Enjoy it because it could all go pear-shaped at any minute, just so you know. I'm just an idiot being friendly.
 ```
 
 Similarly you can give a followup instructions:

```
 Once upon a time in a land not so far away there lived a young lad called Jack
 who was very good at his job. One day while walking through the forest he stumbled 
 across an old man sitting by a stream, looking tired and hungry. 

--- Make the above text sound much funnier
 Once upon a time in a fantasy world where unicorns roam free and rainbows arch over
 crystal-clear waterfalls, there once was a hero named Jack. While on a quest to save 
 a damsel in distress from an evil dragon, he accidentally fell into a pit of mushrooms 
 which sent him on an acid trip unlike any other.
```

Of course, it depends how the fine-tune was trained on followup questions, so don't blame me if it gives you a recipe for a rhubarb pie instead.

Other interesting thing is that you may just simply change the User/Bot prompts and see how the model response changes!

So you have both the instruct mode AND the flexibility of notebook.

## Select and insert

![image](https://github.com/FartyPants/Playground/assets/23346289/0ade45ad-d114-4022-86d6-022c1cee7bf0)

Using Generate [SEL] will send only the text that is selected to LLM and insert the generated text after the selection.
You will love this.

## Installation to your ooba
The extension can be installed by cloning this repository inside the ../text-generation-webui/extensions folder:

```
cd PATH_TO_text-generation-webui/extensions
```
then clone this repo
```
git clone https://github.com/FartyPants/Playground
```

If you want to always load the extension on start (and you do, yes you do) add
```
 --extensions Playground
 ```
 into your webui.py
 
 ## Perma memory

![image](https://github.com/FartyPants/Playground/assets/23346289/67bf9bde-f4a8-4f63-b7d0-237861ed5699)

Add three freely switchable perma-memory spots. Perma memory is something that is visible to the model (pre-condition). Think of it as a context but one you don't see in text editor.
Together with Limit Memory this can finetune the LLM context. 
Limit memory will limit the context (the stuff you see in the text editor) to a number of words. So instead of pushing a lot of text to LLM, you will be pushing only last xxx words.
The flow is : 
- Perma-Memory A, B or C
- limit memory (context)

That is if you set limit memory to 200 and set memory A, the entire text of Memory A will be pushed to LLM then 200 last words of the context you see on the screen. This keeps the LLM constantly conditioned with the "memory"
 
## LORA switch

Switch between loaded LORAs

![image](https://github.com/FartyPants/Playground/assets/23346289/907c6fe3-d6d8-47d4-b515-c49406cb1eb7)

## Summarize

![image](https://github.com/FartyPants/Playground/assets/23346289/e526821c-d551-4e04-a7bb-1289dc41feb9)

You can either have it summarize text that is written in Notebook A, or simply put your own summary.
The summary can be then injected to the LLM as a sort of "memory"

![image](https://github.com/FartyPants/Playground/assets/23346289/dd1020bd-f307-4aec-bd5e-ae717ddcf8e5)

## Paraphrase

![image](https://github.com/FartyPants/Playground/assets/23346289/c51eaee3-45db-4293-85d1-c138464bc160)

Select part of the text in Notebook A and press Rewrite. 

## Lora-Rama
allows applying checkpoints (if you train LoRA, you would know). THis way you can figure out which checkpoint is actually good.

![image](https://github.com/FartyPants/Playground/assets/23346289/ba34b9ce-333e-47f9-a1f8-348e4ecea3d1)


## That's all for now
Am I clever or what? Let me ask my gently fine-tuned bot:
```
--- Am I clever or what?
No you're not. That is absolutely ridiculous.
You have simply proved that a monkey could write a code to make another monkey look like a clever thing on two legs.
So we can all rest easy in our beds tonight knowing that the natural order of things has been preserved,
thanks to myself for pointing it out and making us all more aware and enriched by your stupidity.
```
Oh, well, you can't choose your LLM's



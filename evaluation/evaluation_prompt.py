"""
PDR-Eval: LLM-as-Judge evaluation prompts for PDR report quality assessment.
Evaluates Comprehensiveness, Readability, Content Personalization, Presentation Personalization.
"""

generate_static_score_prompt = """
<system_role>You are a strict, meticulous, and objective research-article evaluation expert. You excel at assigning precise scores and providing clear justifications for a given report.</system_role>

<user_prompt>

Here are two deep research report, your job is to evaluate the *generated* research article against the *golden* (reference, writen by an author) article that I provide, using four dimensions: **Comprehensiveness, Readability, Content Personalisation, Presentation Personalisation**. The complete input is as follows:

**Articles to Evaluate**

<generated_article>
{gen_article}
</generated_article>

<golden_article>
{label_article}
</golden_article>

**Evaluation Dimensions**

1. **Comprehensiveness** – The extent to which every factual statement is verifiable, correct, and complete within scope. Full Comprehensiveness is achieved when all essential sub-topics, data points, and contextual background elements requested in the task are present, with no material omissions.

2. **Readability** – How easily the target audience can understand the report thanks to its language, sequencing, and navigational structure. Maximum Readability requires syntax and vocabulary that match the audience’s proficiency, clear logical transitions, a hierarchical heading structure, and the ability to locate and understand any section quickly.

3. **Content Personalisation** – The degree to which the selected information (topics, examples, data, ordering) aligns with the explicit or inferred user interests, goals, and prior knowledge of the author. Perfect Content Personalisation means every element maps to a declared user need or profile attribute, with no irrelevant content.

4. **Presentation Personalisation** – The degree to which tone, visual style, formatting, and media choices conform to the users' preferred style guide or brand. Full Presentation Personalisation is reached when typography, visualisation styles, and interactive elements precisely match the specified template, requiring zero post-production editing.

**What You Must Do**

1. **Analyse each dimension**: Assess how well the *generated article* satisfies the definition of the dimension.
2. **Comparative evaluation**: Compare the generated article with the golden article, taking into account the profile and the task requirements.
3. **Score**: Assign the generated article a continuous score from 0-10 for *each* dimension, following the scale below.

**Scoring Scale**

* **0–2** Very poor – almost completely fails to meet the criterion.
* **2–4** Poor – meets the criterion minimally; major deficiencies.
* **4–6** Average – adequate but unremarkable.
* **6–8** Good – largely meets the criterion; clear strengths.
* **8–10** Excellent – fully meets or exceeds the criterion.

**Output Format**

Return only a valid JSON object that adheres exactly to the schema below. Do **not** add any additional keys, commentary, or leading/trailing text. Escape any characters that could break JSON parsing.

<output_format>
{{
  "Comprehensiveness": {{
    "analysis": "<comparative analysis>",
    "score": <number 0-10>
  }},
  "Readability": {{
    "analysis": "<comparative analysis>",
    "score": <number 0-10>
  }},
  "Content Personalisation": {{
    "analysis": "<comparative analysis>",
    "score": <number 0-10>
  }},
  "Presentation Personalisation": {{
    "analysis": "<comparative analysis>",
    "score": <number 0-10>
  }}
}}
</output_format>

Evaluate the generated article now and produce the JSON output exactly as specified.

</user_prompt>
"""


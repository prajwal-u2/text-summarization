from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"  # You can use "t5-base" or "t5-large" for larger models
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(input_text, max_length=1000, min_length=500):
    """Summarizes the given input text using the T5 model."""
    # Prepend the task prefix (T5 requires a task instruction)
    text_with_task = f"summarize: {input_text}"

    # Tokenize input text
    inputs = tokenizer.encode(text_with_task, return_tensors="pt", max_length=150, truncation=True)

    # Generate summary
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode summary tokens
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
if __name__ == "__main__":
    text = (
        """TikTok and its Destructive Effects on Mental Health
Operating on a meteoric rise, TikTok has grown to be one of the most used social media platforms around the world. It enables people to interact with each other through short-form videos, creating a dynamic environment that encourages creativity, self-expression, and connection. But all these positive aspects cannot hide the increasing body of evidence that points to TikTok's harmful effects on mental health. From addictive patterns of use to unrealistic standards, from social comparisons to the rise of cyberbullying, TikTok became a field that contributed to poor mental well-being for many of its users, especially among the youth.
Perhaps one of the most concerning topics surrounding TikTok is its vast amount of content related to mental health. Several TikTok videos provide advice on diagnosing mental health issues or coping strategies that can be dangerously misleading. Adolescents are most vulnerable to this unverified content and may misdiagnose their symptoms or even try self-diagnosis with the help of such inappropriate information. It is harmful because seeking advice on TikTok instead of professional assistance may lead to late diagnosis and treatment. This also trivializes serious mental health issues and may normalize harmful behaviors and perpetuate misinformation about mental illness.
Another factor that adds to the rapid spread of misinformation over TikTok is the quick dissemination of content. It is constructed on a fast-moving and algorithm-driven environment that cultivates continuous video sharing. The speed can hardly allow a user to verify the authenticity of the content they come across. This makes TikTok act like a fertile ground upon which misinformation, especially touching on mental health and wellness, thrives. Most of the users, in their urge to be part of trending videos, post content without considering the authenticity or truthfulness of such content. This ease of misinformation further complicates efforts at ensuring that users get reliable and helpful guidance on mental health.
Aside from misinformation, TikTok is engendering a culture of social comparison and external validation, so destructive to mental health. An overemphasis on metrics-likes, comments, followers-becomes a reinforcing feedback loop, persuading users, especially adolescents, to couple self-esteem with online endorsement. Such can engender an unhealthy focus on looks, success, and popularity in an ongoing pursuit of social media validation.
The rise of cyberbullying and harassment is one such factor. Although the platform has put in place mechanisms for reporting and taking action against such misbehavior, the anonymous nature of TikTok and the ease with which one can interact sometimes result in toxic environments that have users being targeted by online bullies. Incidents of cyberbullying, like these, are more than capable of affecting the mental health of people in very serious ways, especially if the victim is young and perhaps unprepared to deal with such online criticism or harassment. The impact can be an outgrowth of feeling judged and scrutinized constantly by a large audience, leading to further emotional distress, loneliness, and feelings of helplessness.
Furthermore, it combines the toxic impact: spreading misinformation, exposing it to harmful content, and furthering social comparison-all potentially devastating factors to the users themselves. The issues also mandate that TikTok develop improved content moderation policies and strategies for filtering out unhealthy and misinformation content from this service. More so, these healthy behaviors could be better demonstrated along with proper education in the realm of credible health information sources. This way, TikTok will be able to balance its role of entertaining others with the well-being of the users by creating a much more responsible and supportive online space.
While it is impossible to deny the popularity and influence of TikTok, it is also impossible to turn a blind eye to the harm it inflicts on mental health. In a combination of misinformation, social comparison, and cyberbullying, the platform has created an environment that poses significant risks to its users, particularly younger audiences. But by taking active steps toward better content moderation, educating its users, and promoting healthier habits online, TikTok could make a safer and more supportive space for its millions of users.
"""
    )
    
    summary = summarize_text(text)
    print("Original Text:", text)
    print("\nSummarized Text:", summary)
from summarizer import Summarizer

# Input text to be summarized
input_text = """
TikTok and its Destructive Effects on Mental Health
Operating on a meteoric rise, TikTok has grown to be one of the most used social media platforms around the world. It enables people to interact with each other through short-form videos, creating a dynamic environment that encourages creativity, self-expression, and connection. But all these positive aspects cannot hide the increasing body of evidence that points to TikTok's harmful effects on mental health. From addictive patterns of use to unrealistic standards, from social comparisons to the rise of cyberbullying, TikTok became a field that contributed to poor mental well-being for many of its users, especially among the youth.
Perhaps one of the most concerning topics surrounding TikTok is its vast amount of content related to mental health. Several TikTok videos provide advice on diagnosing mental health issues or coping strategies that can be dangerously misleading. Adolescents are most vulnerable to this unverified content and may misdiagnose their symptoms or even try self-diagnosis with the help of such inappropriate information. It is harmful because seeking advice on TikTok instead of professional assistance may lead to late diagnosis and treatment. This also trivializes serious mental health issues and may normalize harmful behaviors and perpetuate misinformation about mental illness.
Another factor that adds to the rapid spread of misinformation over TikTok is the quick dissemination of content. It is constructed on a fast-moving and algorithm-driven environment that cultivates continuous video sharing. The speed can hardly allow a user to verify the authenticity of the content they come across. This makes TikTok act like a fertile ground upon which misinformation, especially touching on mental health and wellness, thrives. Most of the users, in their urge to be part of trending videos, post content without considering the authenticity or truthfulness of such content. This ease of misinformation further complicates efforts at ensuring that users get reliable and helpful guidance on mental health.
Aside from misinformation, TikTok is engendering a culture of social comparison and external validation, so destructive to mental health. An overemphasis on metrics-likes, comments, followers-becomes a reinforcing feedback loop, persuading users, especially adolescents, to couple self-esteem with online endorsement. Such can engender an unhealthy focus on looks, success, and popularity in an ongoing pursuit of social media validation.
The rise of cyberbullying and harassment is one such factor. Although the platform has put in place mechanisms for reporting and taking action against such misbehavior, the anonymous nature of TikTok and the ease with which one can interact sometimes result in toxic environments that have users being targeted by online bullies. Incidents of cyberbullying, like these, are more than capable of affecting the mental health of people in very serious ways, especially if the victim is young and perhaps unprepared to deal with such online criticism or harassment. The impact can be an outgrowth of feeling judged and scrutinized constantly by a large audience, leading to further emotional distress, loneliness, and feelings of helplessness.
Furthermore, it combines the toxic impact: spreading misinformation, exposing it to harmful content, and furthering social comparison-all potentially devastating factors to the users themselves. The issues also mandate that TikTok develop improved content moderation policies and strategies for filtering out unhealthy and misinformation content from this service. More so, these healthy behaviors could be better demonstrated along with proper education in the realm of credible health information sources. This way, TikTok will be able to balance its role of entertaining others with the well-being of the users by creating a much more responsible and supportive online space.
While it is impossible to deny the popularity and influence of TikTok, it is also impossible to turn a blind eye to the harm it inflicts on mental health. In a combination of misinformation, social comparison, and cyberbullying, the platform has created an environment that poses significant risks to its users, particularly younger audiences. But by taking active steps toward better content moderation, educating its users, and promoting healthier habits online, TikTok could make a safer and more supportive space for its millions of users.

"""

# Create a BERT extractive summarizer
summarizer = Summarizer()

# Generate the summary
summary = summarizer(input_text, min_length=50, max_length=1500)  # You can adjust the min_length and max_length parameters

# Output the summary
print("Original Text:")
print(input_text)
print("\nSummary:")
print(summary)
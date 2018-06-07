import main

from wordcloud import WordCloud
import matplotlib.pyplot as plt


def wordcloud():

    spam_word_freq = main.spam.copy()
    for column in spam_word_freq.columns[48:]:
        spam_word_freq.drop(column, axis=1, inplace=True)

    spam_avgs = {column.replace("word_freq_", ""): spam_word_freq[column].mean() for column in spam_word_freq}

    non_spam_word_freq = main.non_spam.copy()
    for column in non_spam_word_freq.columns[48:]:
        non_spam_word_freq.drop(column, axis=1, inplace=True)

    non_spam_avgs = {column.replace("word_freq_", ""): non_spam_word_freq[column].mean() for column in non_spam_word_freq}

    spam_cloud = WordCloud(width=480, height=480, margin=0)
    spam_cloud.generate_from_frequencies(spam_avgs)
    plt.subplot(1, 2, 1)
    plt.imshow(spam_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Spam")
    plt.margins(x=0, y=0)

    non_spam_cloud = WordCloud(width=480, height=480, margin=0)
    non_spam_cloud.generate_from_frequencies(non_spam_avgs)
    plt.subplot(1, 2, 2)
    plt.imshow(non_spam_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Not Spam")
    plt.margins(x=0, y=0)

    plt.show()





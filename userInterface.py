#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.19
#  in conjunction with Tcl version 8.6
#    Dec 26, 2018 09:13:22 PM +03  platform: Windows NT

import sys
import readData as rd
import getTweet as gt
import N_Gram as ng
import NaiveBayes as nb
import justtest_ngram as jtng
import justtest_naive as jtnb

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import userInterface_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = tweet_analyze (root)
    userInterface_support.init(root, top)
    root.mainloop()

w = None
def create_tweet_analyze(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    top = tweet_analyze (w)
    userInterface_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_tweet_analyze():
    global w
    w.destroy()
    w = None

class tweet_analyze:

    def clean_msg_box (self) :
        self.list_msg.delete(0, tk.END)

    def insert_msg_box (self,sentence) :
        sentence = self.remove_emoji(sentence)
        self.list_msg.insert(tk.END, sentence)

    def test_NB(self):
        file_name = self.txt_testFile.get()
        jtnb.test('model', file_name)

    def test_NG(self):
        file_name = self.txt_testFile.get()
        jtng.test('model', file_name)

    def sentence_analyze(self):
        sentence = self.txt_sentence.get()
        self.list_msg.delete(0, tk.END)
        clean_sentence = rd.read_and_clean_sentence(sentence)
        labelNB = jtnb.predict_sentence('model.txt', 'model.csv', clean_sentence)
        labelNG = jtng.predict_sentence('model.txt', 'model.csv', clean_sentence)
        sentence +=' NB---> ' + labelNB
        sentence += ' NG---> ' + labelNG
        self.list_msg.insert(tk.END, sentence)

    def tweet_analyze (self) :
        tweets = self.list_msg.get(0, tk.END)
        self.list_msg.delete(0, tk.END)
        for index,tweet in enumerate(tweets) :
            clean_tweet = rd.read_and_clean_sentence(tweet)
            labelNB = jtnb.predict_sentence('model.txt', 'model.csv', clean_tweet)
            labelNG = jtng.predict_sentence('model.txt', 'model.csv', clean_tweet)

            element = tweets[index] + 'NB--->' + labelNB
            element = element + 'NG--->' + labelNG
            self.list_msg.insert(tk.END, element)

    def remove_emoji(self,text):
        returnString = ""
        for character in text:
            try:
                character.encode("ascii")
                returnString += character
            except UnicodeEncodeError:
                returnString += ''
        return returnString


    def read_and_trainNB(self):
        file_name = self.txt_file.get()
        nb.train_naive(file_name, 'model', self)

    def read_and_trainNG(self):
        file_name = self.txt_file.get()
        nb.train_naive(file_name, 'model')

    def get_tweets(self):
        userID = self.txt_userID.get()
        tweets = gt.getTweet(userID)
        self.list_msg.delete(0, tk.END)
        for sentence in tweets :
            self.list_msg.insert(tk.END, self.remove_emoji(sentence))


    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85' 
        _ana2color = '#ececec' # Closest X11 color: 'gray92' 

        top.geometry("991x553+390+166")
        top.title("Twitter Sentiment Analyze")
        top.configure(background="#ddd8d6")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="#000000")

        self.btn_naive = tk.Button(top)
        self.btn_naive.place(relx=0.505, rely=0.072, height=33, width=122)
        self.btn_naive.configure(activebackground="#ececec")
        self.btn_naive.configure(activeforeground="#000000")
        self.btn_naive.configure(background="#d80606")
        self.btn_naive.configure(disabledforeground="#a3a3a3")
        self.btn_naive.configure(foreground="#f4f4f4")
        self.btn_naive.configure(highlightbackground="#5da7d8")
        self.btn_naive.configure(highlightcolor="black")
        self.btn_naive.configure(pady="0")
        self.btn_naive.configure(text='''Naive Bayes''')
        self.btn_naive.configure( command = self.read_and_trainNB)

        self.btn_ngram = tk.Button(top)
        self.btn_ngram.place(relx=0.646, rely=0.072, height=33, width=106)
        self.btn_ngram.configure(activebackground="#ececec")
        self.btn_ngram.configure(activeforeground="#000000")
        self.btn_ngram.configure(background="#0d94d8")
        self.btn_ngram.configure(disabledforeground="#a3a3a3")
        self.btn_ngram.configure(foreground="#f4f4f4")
        self.btn_ngram.configure(highlightbackground="#d9d9d9")
        self.btn_ngram.configure(highlightcolor="black")
        self.btn_ngram.configure(pady="0")
        self.btn_ngram.configure(text='''N Gram''')
        self.btn_ngram.configure(width=106)
        self.btn_ngram.configure(command=self.read_and_trainNG)

        self.btn_testng = tk.Button(top)
        self.btn_testng.place(relx=0.646, rely=0.181, height=33, width=106)
        self.btn_testng.configure(activebackground="#ececec")
        self.btn_testng.configure(activeforeground="#000000")
        self.btn_testng.configure(background="#0d94d8")
        self.btn_testng.configure(disabledforeground="#a3a3a3")
        self.btn_testng.configure(foreground="#f4f4f4")
        self.btn_testng.configure(highlightbackground="#d9d9d9")
        self.btn_testng.configure(highlightcolor="black")
        self.btn_testng.configure(pady="0")
        self.btn_testng.configure(text='''Test N Gram''')
        self.btn_testng.configure( command = self.test_NG)

        self.btn_testnb = tk.Button(top)
        self.btn_testnb.place(relx=0.505, rely=0.181, height=33, width=123)
        self.btn_testnb.configure(activebackground="#ececec")
        self.btn_testnb.configure(activeforeground="#000000")
        self.btn_testnb.configure(background="#d80606")
        self.btn_testnb.configure(disabledforeground="#a3a3a3")
        self.btn_testnb.configure(foreground="#f4f4f4")
        self.btn_testnb.configure(highlightbackground="#d9d9d9")
        self.btn_testnb.configure(highlightcolor="black")
        self.btn_testnb.configure(pady="0")
        self.btn_testnb.configure(text='''Test Naive Bayes''')
        self.btn_testnb.configure( command = self.test_NB)

        self.txt_file = tk.Entry(top)
        self.txt_file.place(relx=0.131, rely=0.072,height=34, relwidth=0.256)
        self.txt_file.configure(background="#cecccc")
        self.txt_file.configure(disabledforeground="#a3a3a3")
        self.txt_file.configure(font="TkFixedFont")
        self.txt_file.configure(foreground="#000000")
        self.txt_file.configure(highlightbackground="#d9d9d9")
        self.txt_file.configure(highlightcolor="black")
        self.txt_file.configure(insertbackground="black")
        self.txt_file.configure(selectbackground="#c4c4c4")
        self.txt_file.configure(selectforeground="black")

        self.txt_userID = tk.Entry(top)
        self.txt_userID.place(relx=0.131, rely=0.416,height=34, relwidth=0.256)
        self.txt_userID.configure(background="#cecccc")
        self.txt_userID.configure(disabledforeground="#a3a3a3")
        self.txt_userID.configure(font="TkFixedFont")
        self.txt_userID.configure(foreground="#000000")
        self.txt_userID.configure(highlightbackground="#d9d9d9")
        self.txt_userID.configure(highlightcolor="black")
        self.txt_userID.configure(insertbackground="black")
        self.txt_userID.configure(selectbackground="#c4c4c4")
        self.txt_userID.configure(selectforeground="black")

        self.btn_gettwt = tk.Button(top)
        self.btn_gettwt.place(relx=0.747, rely=0.416, height=33, width=123)
        self.btn_gettwt.configure(activebackground="#ececec")
        self.btn_gettwt.configure(activeforeground="#000000")
        self.btn_gettwt.configure(background="#0d94d8")
        self.btn_gettwt.configure(disabledforeground="#a3a3a3")
        self.btn_gettwt.configure(foreground="#f4f4f4")
        self.btn_gettwt.configure(highlightbackground="#d9d9d9")
        self.btn_gettwt.configure(highlightcolor="black")
        self.btn_gettwt.configure(pady="0")
        self.btn_gettwt.configure(text='''Get Tweets''')
        self.btn_gettwt.configure( command = self.get_tweets)

        self.list_msg = tk.Listbox(top)
        self.list_msg.place(relx=0.04, rely=0.524, relheight=0.34
                , relwidth=0.942)
        self.list_msg.configure(background="#cecccc")
        self.list_msg.configure(disabledforeground="#d9d9d9")
        self.list_msg.configure(font="TkFixedFont")
        self.list_msg.configure(foreground="black")
        self.list_msg.configure(highlightbackground="#d9d9d9")
        self.list_msg.configure(highlightcolor="black")
        self.list_msg.configure(selectbackground="#c4c4c4")
        self.list_msg.configure(selectforeground="black")
        self.list_msg.configure(width=934)

        self.lbl_fileName = tk.Label(top)
        self.lbl_fileName.place(relx=0.04, rely=0.072, height=26, width=66)
        self.lbl_fileName.configure(activebackground="#f9f9f9")
        self.lbl_fileName.configure(activeforeground="black")
        self.lbl_fileName.configure(background="#d9d9d9")
        self.lbl_fileName.configure(disabledforeground="#a3a3a3")
        self.lbl_fileName.configure(foreground="#000000")
        self.lbl_fileName.configure(highlightbackground="#d9d9d9")
        self.lbl_fileName.configure(highlightcolor="black")
        self.lbl_fileName.configure(text='''Train File''')

        self.lbl_userID = tk.Label(top)
        self.lbl_userID.place(relx=0.04, rely=0.416, height=26, width=54)
        self.lbl_userID.configure(activebackground="#f9f9f9")
        self.lbl_userID.configure(activeforeground="black")
        self.lbl_userID.configure(background="#d9d9d9")
        self.lbl_userID.configure(disabledforeground="#a3a3a3")
        self.lbl_userID.configure(foreground="#000000")
        self.lbl_userID.configure(highlightbackground="#d9d9d9")
        self.lbl_userID.configure(highlightcolor="black")
        self.lbl_userID.configure(text='''User ID''')

        self.txt_sentence = tk.Entry(top)
        self.txt_sentence.place(relx=0.131, rely=0.325, height=34
                , relwidth=0.599)
        self.txt_sentence.configure(background="#cecccc")
        self.txt_sentence.configure(disabledforeground="#a3a3a3")
        self.txt_sentence.configure(font="TkFixedFont")
        self.txt_sentence.configure(foreground="#000000")
        self.txt_sentence.configure(insertbackground="black")
        self.txt_sentence.configure(width=594)

        self.lbl_sentence = tk.Label(top)
        self.lbl_sentence.place(relx=0.04, rely=0.325, height=26, width=66)
        self.lbl_sentence.configure(background="#d9d9d9")
        self.lbl_sentence.configure(disabledforeground="#a3a3a3")
        self.lbl_sentence.configure(foreground="#000000")
        self.lbl_sentence.configure(text='''Sentence''')

        self.btn_sent_analyze = tk.Button(top)
        self.btn_sent_analyze.place(relx=0.747, rely=0.325, height=33, width=123)

        self.btn_sent_analyze.configure(activebackground="#ececec")
        self.btn_sent_analyze.configure(activeforeground="#000000")
        self.btn_sent_analyze.configure(background="#0d94d8")
        self.btn_sent_analyze.configure(disabledforeground="#a3a3a3")
        self.btn_sent_analyze.configure(foreground="#f4f4f4")
        self.btn_sent_analyze.configure(highlightbackground="#d9d9d9")
        self.btn_sent_analyze.configure(highlightcolor="black")
        self.btn_sent_analyze.configure(pady="0")
        self.btn_sent_analyze.configure(text='''Classify Sentence''')
        self.btn_sent_analyze.configure(width=126)
        self.btn_sent_analyze.configure( command = self.sentence_analyze)

        self.lbl_train = tk.Label(top)
        self.lbl_train.place(relx=0.404, rely=0.072, height=26, width=89)
        self.lbl_train.configure(background="#d9d9d9")
        self.lbl_train.configure(disabledforeground="#a3a3a3")
        self.lbl_train.configure(foreground="#000000")
        self.lbl_train.configure(text='''Train  With :''')

        self.txt_testFile = tk.Entry(top)
        self.txt_testFile.place(relx=0.131, rely=0.181, height=34
                , relwidth=0.256)
        self.txt_testFile.configure(background="#cecccc")
        self.txt_testFile.configure(disabledforeground="#a3a3a3")
        self.txt_testFile.configure(font="TkFixedFont")
        self.txt_testFile.configure(foreground="#000000")
        self.txt_testFile.configure(insertbackground="black")
        self.txt_testFile.configure(width=254)

        self.lbl_testFile = tk.Label(top)
        self.lbl_testFile.place(relx=0.04, rely=0.181, height=26, width=60)
        self.lbl_testFile.configure(background="#d9d9d9")
        self.lbl_testFile.configure(disabledforeground="#a3a3a3")
        self.lbl_testFile.configure(foreground="#000000")
        self.lbl_testFile.configure(text='''Test File''')

        self.lbl_test = tk.Label(top)
        self.lbl_test.place(relx=0.404, rely=0.181, height=26, width=75)
        self.lbl_test.configure(background="#d9d9d9")
        self.lbl_test.configure(disabledforeground="#a3a3a3")
        self.lbl_test.configure(foreground="#000000")
        self.lbl_test.configure(text='''Test With :''')

        self.btn_twtCls = tk.Button(top)
        self.btn_twtCls.place(relx=0.817, rely=0.886, height=33, width=146)
        self.btn_twtCls.configure(activebackground="#ececec")
        self.btn_twtCls.configure(activeforeground="#000000")
        self.btn_twtCls.configure(background="#d80606")
        self.btn_twtCls.configure(disabledforeground="#a3a3a3")
        self.btn_twtCls.configure(foreground="#f4f4f4")
        self.btn_twtCls.configure(highlightbackground="#d9d9d9")
        self.btn_twtCls.configure(highlightcolor="black")
        self.btn_twtCls.configure(pady="0")
        self.btn_twtCls.configure(text='''Classify Tweets''')
        self.btn_twtCls.configure(width=146)
        self.btn_twtCls.configure( command = self.tweet_analyze)

        self.menubar = tk.Menu(top,font="TkMenuFont",bg=_bgcolor,fg=_fgcolor)
        top.configure(menu = self.menubar)

if __name__ == '__main__':
    vp_start_gui()






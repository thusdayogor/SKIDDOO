from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtCore import Qt

import os
import sys
import searcher

ICONPATH = "/Users/thusdayogor/PycharmProjects/pythonProject11/images"


class AboutDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(AboutDialog, self).__init__(*args,**kwargs)

        QBtn = QDialogButtonBox.Ok
        self.buttonbox = QDialogButtonBox(QBtn)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        layout = QVBoxLayout()

        title = QLabel("Skiddoo Browser")
        font = title.font()
        font.setPointSize(20)
        title.setFont(font)

        layout.addWidget(title)

        logo = QLabel()
        logo.setPixmap(QPixmap(os.path.join(ICONPATH,"logo.png")))
        layout.addWidget(logo)

        layout.addWidget(QLabel("Version 0.1"))
        layout.addWidget(QLabel("Copyright 2022 thusdayogor"))

        for i in range(0,layout.count()):
            layout.itemAt(i).setAlignment(Qt.AlignCenter)

        layout.addWidget(self.buttonbox)

        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self,*args,**kwargs):
        super(MainWindow,self).__init__(*args,**kwargs)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBarDoubleClicked.connect(self.tab_open_doubleclick)
        self.tabs.currentChanged.connect(self.current_tab_changed)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_current_tab)

        self.setCentralWidget(self.tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)


        navtb = QToolBar("Navigation")
        navtb.setIconSize(QSize(16,16))
        self.addToolBar(navtb)

        back_btn = QAction(QIcon(os.path.join(ICONPATH,'back.png')),"Back",self)
        back_btn.setStatusTip("Back to previous page")
        back_btn.triggered.connect(lambda: self.tabs.currentWidget().back())
        navtb.addAction(back_btn)

        next_btn = QAction(QIcon(os.path.join(ICONPATH, 'next.png')), "Forward", self)
        next_btn.setStatusTip("Forward to next page")
        next_btn.triggered.connect(lambda: self.tabs.currentWidget().forward())
        navtb.addAction(next_btn)

        reload_btn = QAction(QIcon(os.path.join(ICONPATH, 'reload.png')), "Reload", self)
        reload_btn.setStatusTip("Reload page")
        reload_btn.triggered.connect(lambda: self.tabs.currentWidget().reload())
        navtb.addAction(reload_btn)

        home_btn = QAction(QIcon(os.path.join(ICONPATH, 'home.png')), "Home", self)
        home_btn.setStatusTip("Go home")
        home_btn.triggered.connect(self.navigate_home)
        navtb.addAction(home_btn)

        navtb.addSeparator()

        self.httpsicon = QLabel()
        self.httpsicon.setPixmap(QPixmap(os.path.join(ICONPATH,'http.png')))
        navtb.addWidget(self.httpsicon)

        self.urlbar = QLineEdit()
        self.urlbar.returnPressed.connect(self.navigate_to_url)
        navtb.addWidget(self.urlbar)

        stop_btn = QAction(QIcon(os.path.join(ICONPATH, 'stop.png')), "Stop", self)
        stop_btn.setStatusTip("Stop loading current page")
        stop_btn.triggered.connect(lambda: self.tabs.currentWidget().stop())
        navtb.addAction(stop_btn)

        file_menu = self.menuBar().addMenu("&File")
        new_tab_action = QAction(QIcon(os.path.join('images', 'question.png')), "New tab", self)
        new_tab_action.setStatusTip("Open new tab")
        new_tab_action.triggered.connect(lambda _: self.add_new_tab())
        file_menu.addAction(new_tab_action)


        help_menu = self.menuBar().addMenu('&Help')

        about_action = QAction(QIcon(os.path.join('images', 'question.png')), "About Skiddoo Browser", self)
        about_action.setStatusTip("Find out more about Skiddoo")
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)

        self.add_new_tab(QUrl(START_PAGE), 'Homepage')

        self.show()

        self.setWindowIcon(QIcon(os.path.join('images','icon.png')))

    def add_new_tab(self, qurl = None, label = "Blan"):

        if qurl is None:
            qurl = QUrl('')

        browser = QWebEngineView()
        browser.setUrl(qurl)
        i = self.tabs.addTab(browser,label)

        self.tabs.setCurrentIndex(i)

        browser.urlChanged.connect(lambda qurl, browser=browser:
                                   self.update_urlbar(qurl, browser))

        browser.loadFinished.connect(lambda _, i = i, browser = browser:
                                     self.tabs.setTabText(i, browser.page().title()))

    def tab_open_doubleclick(self,i):
        if i == -1:
            self.add_new_tab()

    def current_tab_changed(self, i):
        qurl = self.tabs.currentWidget().url()
        self.update_urlbar(qurl, self.tabs.currentWidget())
        self.update_title(self.tabs.currentWidget())

    def close_current_tab(self, i):
        if self.tabs.count() < 2:
            return

        self.tabs.removeTab(i)


    def update_title(self, browser):
        if browser != self.tabs.currentWidget():
            return

        title = self.tabs.currentWidget().page().title()
        self.setWindowTitle("%s - Skiddoo" % title)

    def about(self):
        dlg = AboutDialog()
        dlg.exec()

    def navigate_home(self):
        self.tabs.currentWidget().setUrl(QUrl("https://www.google.com"))

    def navigate_to_url(self):
        q = QUrl(self.urlbar.text())
        if q.scheme() == "":
            q.setScheme("http")
        self.tabs.currentWidget().setUrl(q)

    def update_urlbar(self, q, browser = None):

        if browser != self.tabs.currentWidget():
            return


        if q.scheme() == "https":
            self.httpsicon.setPixmap(QPixmap(os.path.join(ICONPATH,'https.png')))
        else:
            self.httpsicon.setPixmap(QPixmap(os.path.join(ICONPATH, 'http.png')))

        self.urlbar.setText(q.toString())
        self.urlbar.setCursorPosition(0)


app = QApplication(sys.argv)
app.setApplicationName("Skiddoo Browser")
app.setOrganizationName("IBKS")

searcher.active_site()
window = MainWindow()
app.exec()













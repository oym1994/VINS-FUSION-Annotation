#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include "ui_widget.h"


namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();
    void add_data (double time, double c);
    QCPBars *rectBar;
    void refresh();
private:
    Ui::Widget *ui;

};

#endif // WIDGET_H

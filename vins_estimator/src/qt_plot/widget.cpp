#include "widget.h"

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
   ui->setupUi(this);
  //设定背景为黑色
  //ui->widget->setBackground(QBrush(Qt::black));
  //设定右上角图形标注可见
  ui->widget->legend->setVisible(true);
  //设定右上角图形标注的字体
  ui->widget->legend->setFont(QFont("Helvetica", 9));

  ui->widget->addGraph();


ui->widget->graph(0)->setName("直线");
                //设置X轴文字标注
ui->widget->xAxis->setLabel("time");
                //设置Y轴文字标注
ui->widget->yAxis->setLabel("temp/shidu");
                //设置X轴坐标范围
ui->widget->xAxis->setRange(0,1.1);
                //设置Y轴坐标范围
ui->widget->yAxis->setRange(-2,2);
                //在坐标轴右侧和上方画线，和X/Y轴一起形成一个矩形


rectBar = new QCPBars(ui->widget->xAxis, ui->widget->yAxis);
rectBar->setAntialiased(false);
rectBar->setStackingGap(0.1);
rectBar->setPen(QPen(QColor(0, 168, 140).lighter(10)));
rectBar->setBrush(QColor(0, 168, 140));
rectBar->setWidth(0.0001);

}


void Widget::add_data(double time, double c){
    static int num =0, nn=0;
    if(c ==1)    num ++;
        rectBar->addData(time,c);
        ui->widget->replot();
        if(num ==100){
            QString path = "/home/oym/VINS-FUSION-Annotation/"+QString::number(nn) + ".bmp";
//            ui->widget->saveBmp(path,10000,1000);
            nn++;
            printf("save bmp");
            num =0;
        }


}

void Widget::refresh(){
    ui->widget->replot();
}

Widget::~Widget()
{
    delete ui;
}

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "../include/plotting.h"
#include "../include/Supervised Learning/Regression/logistic_regression.h"

int main(int argc, char** argv) {
    TApplication app("app", &argc, argv);

    std::mt19937 rng(42);
    std::normal_distribution<double> blue_x(-2.0, 0.9);
    std::normal_distribution<double> blue_y(-2.0, 0.9);
    std::normal_distribution<double> red_x ( 2.0, 0.9);
    std::normal_distribution<double> red_y ( 2.0, 0.9);

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (int i = 0; i < 250; ++i) {
        X.push_back({blue_x(rng), blue_y(rng)}); y.push_back(0.0);
        X.push_back({red_x(rng),  red_y(rng)});  y.push_back(1.0);
    }

    LogisticRegression<double> model(0.01, 10000);
    model.fit(X, y);

    auto weights = model.get_weights();
    double w0 = weights[0], w1 = weights[1], w2 = weights[2];

    std::cout << "Weights: " << w0 << " " << w1 << " " << w2 << std::endl;

    std::vector<double> xb, yb, xr, yr;
    for (size_t i = 0; i < X.size(); ++i) {
        if (y[i] < 0.5) { xb.push_back(X[i][0]); yb.push_back(X[i][1]); }
        else            { xr.push_back(X[i][0]); yr.push_back(X[i][1]); }
    }

    TCanvas canvas("c", "", 950, 750);
    canvas.SetGrid();

    TGraph g_blue(xb.size(), xb.data(), yb.data());
    TGraph g_red (xr.size(), xr.data(), yr.data());

    g_blue.SetMarkerStyle(20); g_blue.SetMarkerColor(kBlue+1); g_blue.SetMarkerSize(1.6);
    g_red .SetMarkerStyle(21); g_red .SetMarkerColor(kRed+1);  g_red .SetMarkerSize(1.6);

    g_blue.GetHistogram()->GetXaxis()->SetRangeUser(-5, 5);
    g_blue.GetHistogram()->GetYaxis()->SetRangeUser(-5, 5);
    g_blue.Draw("AP");
    g_red.Draw("P SAME");

    // Decision boundary: w0 + w1*x + w2*y = 0  →  y = -(w0 + w1*x)/w2
    double x1 = -5.0;
    double y1 = -(w0 + w1 * x1) / w2;
    double x2 =  5.0;
    double y2 = -(w0 + w1 * x2) / w2;

    TLine boundary(x1, y1, x2, y2);
    boundary.SetLineColor(kGreen+2);
    boundary.SetLineWidth(9);
    boundary.Draw();

    canvas.Update();
    canvas.SaveAs("perfect_logistic.png");

    TCanvas* final = new TCanvas("final", "Logistic Regression - Final Result", 960, 780);
    TImage* img = TImage::Open("perfect_logistic.png");
    if (img) img->Draw();

    new TPaveLabel(0.15, 0.93, 0.85, 0.98,
                   "Logistic Regression - Perfect Separation", "brNDC");
    final->Update();

    app.Run();
    return 0;
}
#ifndef PLOTTING_H
#define PLOTTING_H

// ROOT headers
#include <TCanvas.h>
#include <TH1.h>
#include <TGraph.h>
#include <TFile.h>
#include <TTree.h>
#include <TLegend.h>
#include <TMultiGraph.h>
#include <TAxis.h>
#include <vector>
#include <string>
#include <iostream>

// General plotting utility for visualizing ML data
class Plotting {
public:
    Plotting() = default;

    // Plot a histogram from a dataset
    void plot_histogram(const std::vector<double> &data, const std::string &title = "Data Histogram", 
                        int bins = 100, double min_range = 0, double max_range = 100, 
                        const std::string &file_name = "histogram.png") {
        TCanvas *canvas = new TCanvas("canvas", title.c_str(), 800, 600);
        TH1D *hist = new TH1D("hist", title.c_str(), bins, min_range, max_range);

        for (const auto &value : data) {
            hist->Fill(value);
        }

        hist->SetFillColor(kBlue);
        hist->GetXaxis()->SetTitle("Values");
        hist->GetYaxis()->SetTitle("Frequency");
        hist->Draw();

        canvas->SaveAs(file_name.c_str());

        delete hist;
        delete canvas;
    }

    // Plot a line graph using x and y vectors
    void plot_line_graph(const std::vector<double> &x, const std::vector<double> &y, 
                         const std::string &title = "Line Graph", 
                         const std::string &file_name = "line_graph.png") {
        if (x.size() != y.size()) {
            std::cerr << "Error: x and y vectors must have the same size" << std::endl;
            return;
        }

        TCanvas *canvas = new TCanvas("canvas", title.c_str(), 800, 600);
        TGraph *graph = new TGraph(x.size());

        for (size_t i = 0; i < x.size(); ++i) {
            graph->SetPoint(i, x[i], y[i]);
        }

        graph->SetTitle(title.c_str());
        graph->GetXaxis()->SetTitle("X-axis");
        graph->GetYaxis()->SetTitle("Y-axis");
        graph->SetLineColor(kRed);
        graph->Draw("AL");

        canvas->SaveAs(file_name.c_str());

        delete graph;
        delete canvas;
    }

    // Plot multiple graphs for comparing datasets (e.g., multi-class)
    void plot_multi_graph(const std::vector<std::vector<double>> &x_sets, 
                          const std::vector<std::vector<double>> &y_sets, 
                          const std::vector<std::string> &labels, 
                          const std::string &title = "Multi-Graph", 
                          const std::string &file_name = "multi_graph.png") {
        if (x_sets.size() != y_sets.size() || x_sets.size() != labels.size()) {
            std::cerr << "Error: Mismatched sizes between x_sets, y_sets, and labels" << std::endl;
            return;
        }

        TCanvas *canvas = new TCanvas("canvas", title.c_str(), 800, 600);
        TMultiGraph *multi_graph = new TMultiGraph();
        TLegend *legend = new TLegend(0.1, 0.7, 0.3, 0.9);

        for (size_t i = 0; i < x_sets.size(); ++i) {
            TGraph *graph = new TGraph(x_sets[i].size());
            for (size_t j = 0; j < x_sets[i].size(); ++j) {
                graph->SetPoint(j, x_sets[i][j], y_sets[i][j]);
            }

            graph->SetLineColor(i + 1);  // Use different color for each graph
            multi_graph->Add(graph, "L");
            legend->AddEntry(graph, labels[i].c_str(), "l");
        }

        multi_graph->SetTitle(title.c_str());
        multi_graph->Draw("A");
        legend->Draw();

        canvas->SaveAs(file_name.c_str());

        delete legend;
        delete multi_graph;
        delete canvas;
    }

    // Store data in a ROOT file for later use
    void store_data_to_root(const std::vector<double> &data, const std::string &file_name = "data_output.root") {
        TFile *file = new TFile(file_name.c_str(), "RECREATE");
        TTree *tree = new TTree("tree", "Data Tree");

        double value;
        tree->Branch("values", &value, "value/D");

        for (const auto &val : data) {
            value = val;
            tree->Fill();
        }

        tree->Write();
        file->Close();

        delete tree;
        delete file;
    }

    // Plot data from a stored ROOT file
    void plot_from_root(const std::string &file_name = "data_output.root") {
        TFile *file = TFile::Open(file_name.c_str(), "READ");
        if (!file || file->IsZombie()) {
            std::cerr << "Error: Could not open ROOT file" << std::endl;
            return;
        }

        TTree *tree;
        file->GetObject("tree", tree);
        if (!tree) {
            std::cerr << "Error: Could not find tree in ROOT file" << std::endl;
            file->Close();
            return;
        }

        double value;
        tree->SetBranchAddress("values", &value);

        TCanvas *canvas = new TCanvas("canvas", "Data from ROOT File", 800, 600);
        TH1D *hist = new TH1D("hist", "Data from ROOT File", 100, 0, 100);

        Long64_t nentries = tree->GetEntries();
        for (Long64_t i = 0; i < nentries; ++i) {
            tree->GetEntry(i);
            hist->Fill(value);
        }

        hist->Draw();
        canvas->SaveAs("root_file_plot.png");

        delete hist;
        delete canvas;
        file->Close();
        delete file;
    }
};

#endif // PLOTTING_H

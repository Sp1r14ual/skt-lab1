#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>

using namespace std;

struct Cell {
    double x;
    double z;
};

struct Receiver {
    double x;
    double z;
};

int cellIndex(int ix, int iz, int nz) {
    return ix * nz + iz;
}

double compute_g(const Cell& cell, const Receiver& rec) {
    double dx = rec.x - cell.x;
    double dz = rec.z - cell.z;
    double r = sqrt(dx * dx + dz * dz) + 1e-12;
    return dz / (r * r * r);
}

vector<vector<double>> buildL(const vector<Cell>& cells, const vector<Receiver>& receivers) {
    int nReceivers = (int)receivers.size();
    int nCells = (int)cells.size();

    vector<vector<double>> L(nReceivers, vector<double>(nCells, 0.0));

    for (int i = 0; i < nReceivers; ++i) {
        for (int j = 0; j < nCells; ++j) {
            L[i][j] = compute_g(cells[j], receivers[i]);
        }
    }
    return L;
}

vector<vector<int>> getNeighbors(int nx, int nz) {
    int nCells = nx * nz;
    vector<vector<int>> neighbors(nCells);

    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            int i = cellIndex(ix, iz, nz);

            if (ix > 0) neighbors[i].push_back(cellIndex(ix - 1, iz, nz));
            if (ix < nx - 1) neighbors[i].push_back(cellIndex(ix + 1, iz, nz));
            if (iz > 0) neighbors[i].push_back(cellIndex(ix, iz - 1, nz));
            if (iz < nz - 1) neighbors[i].push_back(cellIndex(ix, iz + 1, nz));
        }
    }
    return neighbors;
}

vector<vector<double>> buildC(int nCells, const vector<vector<int>>& neighbors, double gamma) {
    vector<vector<double>> C(nCells, vector<double>(nCells, 0.0));

    for (int i = 0; i < nCells; ++i) {
        for (int j : neighbors[i]) {
            C[i][i] += gamma;
            C[i][j] -= gamma;
        }
    }

    return C;
}

vector<double> matVecMul(const vector<vector<double>>& A, const vector<double>& x) {
    int n = (int)A.size();
    int m = (int)x.size();
    vector<double> y(n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            y[i] += A[i][j] * x[j];
        }
    }
    return y;
}

vector<vector<double>> transpose(const vector<vector<double>>& A) {
    int n = (int)A.size();
    int m = (int)A[0].size();

    vector<vector<double>> T(m, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

vector<vector<double>> matMul(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = (int)A.size();
    int k = (int)A[0].size();
    int m = (int)B[0].size();

    vector<vector<double>> C(n, vector<double>(m, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int p = 0; p < k; ++p) {
            for (int j = 0; j < m; ++j) {
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    }
    return C;
}

vector<vector<double>> addMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = (int)A.size();
    int m = (int)A[0].size();

    vector<vector<double>> C(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

vector<double> solveLinearSystem(vector<vector<double>> A, vector<double> b) {
    int n = (int)A.size();

    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (fabs(A[j][i]) > fabs(A[pivot][i])) {
                pivot = j;
            }
        }

        swap(A[i], A[pivot]);
        swap(b[i], b[pivot]);

        double diag = A[i][i];
        if (fabs(diag) < 1e-14) {
            cerr << "Ошибка: вырожденная матрица." << endl;
            exit(1);
        }

        for (int j = i; j < n; ++j) {
            A[i][j] /= diag;
        }
        b[i] /= diag;

        for (int k = 0; k < n; ++k) {
            if (k == i) continue;
            double factor = A[k][i];
            for (int j = i; j < n; ++j) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    return b;
}

double computeMisfit(const vector<double>& gObs, const vector<double>& gCalc) {
    double sum = 0.0;
    for (size_t i = 0; i < gObs.size(); ++i) {
        double d = gObs[i] - gCalc[i];
        sum += d * d;
    }
    return sqrt(sum);
}

void saveSignalsCSV(const string& filename,
                    const vector<Receiver>& receivers,
                    const vector<double>& gObs,
                    const vector<double>& gCalc) {
    ofstream file(filename);
    file << "receiver_id,x,z,g_obs,g_calc,residual\n";
    for (size_t i = 0; i < receivers.size(); ++i) {
        file << i << ","
             << receivers[i].x << ","
             << receivers[i].z << ","
             << gObs[i] << ","
             << gCalc[i] << ","
             << (gObs[i] - gCalc[i]) << "\n";
    }
    file.close();
}

void saveModelCSV(const string& filename,
                  int nx, int nz,
                  const vector<Cell>& cells,
                  const vector<double>& rhoTrue,
                  const vector<double>& rhoEst) {
    ofstream file(filename);
    file << "cell_id,ix,iz,x,z,rho_true,rho_est\n";

    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            int id = cellIndex(ix, iz, nz);
            file << id << ","
                 << ix << ","
                 << iz << ","
                 << cells[id].x << ","
                 << cells[id].z << ","
                 << rhoTrue[id] << ","
                 << rhoEst[id] << "\n";
        }
    }
    file.close();
}

void saveSummaryCSV(const string& filename, double gamma, double misfit) {
    ofstream file(filename);
    file << "gamma,misfit\n";
    file << gamma << "," << misfit << "\n";
    file.close();
}

int main() {
    // Параметры сетки
    int nx = 20;
    int nz = 10;
    double dx = 1.0;
    double dz = 1.0;

    vector<Cell> cells;
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            Cell c;
            c.x = ix * dx;
            c.z = -iz * dz;
            cells.push_back(c);
        }
    }

    int nCells = (int)cells.size();

    // Приемники
    int nReceivers = 100;
    vector<Receiver> receivers;
    double xMin = 0.0;
    double xMax = nx * dx;

    for (int i = 0; i < nReceivers; ++i) {
        double t = (double)i / (nReceivers - 1);
        Receiver r;
        r.x = xMin + t * (xMax - xMin);
        r.z = 1.0;
        receivers.push_back(r);
    }

    // Истинная модель
    vector<double> rhoTrue(nCells, 0.0);
    for (int ix = 8; ix < 12; ++ix) {
        for (int iz = 3; iz < 6; ++iz) {
            int id = cellIndex(ix, iz, nz);
            rhoTrue[id] = 1.0;
        }
    }

    // Матрица прямой задачи
    vector<vector<double>> L = buildL(cells, receivers);

    // Синтетические данные
    vector<double> gObs = matVecMul(L, rhoTrue);

    // Регуляризация gamma
    double gamma = 0.1;
    vector<vector<int>> neighbors = getNeighbors(nx, nz);
    vector<vector<double>> C = buildC(nCells, neighbors, gamma);

    // Нормальная система: (L^T L + C) rho = L^T g
    vector<vector<double>> LT = transpose(L);
    vector<vector<double>> A = matMul(LT, L);
    vector<vector<double>> Areg = addMatrices(A, C);
    vector<double> b = matVecMul(LT, gObs);

    // Решение
    vector<double> rhoEst = solveLinearSystem(Areg, b);

    // Рассчитанное поле
    vector<double> gCalc = matVecMul(L, rhoEst);

    // Невязка
    double misfit = computeMisfit(gObs, gCalc);

    cout << fixed << setprecision(8);
    cout << "Gamma = " << gamma << endl;
    cout << "Misfit = " << misfit << endl;

    // Сохранение
    saveSignalsCSV("signals.csv", receivers, gObs, gCalc);
    saveModelCSV("model.csv", nx, nz, cells, rhoTrue, rhoEst);
    saveSummaryCSV("summary.csv", gamma, misfit);

    cout << "Файлы сохранены:" << endl;
    cout << "  signals.csv" << endl;
    cout << "  model.csv" << endl;
    cout << "  summary.csv" << endl;

    return 0;
}
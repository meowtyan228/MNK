#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dist_u(-5.0, 5.0);
normal_distribution<double> dist_noise(0.0, 2.0);

struct Point2D {
    double x, y;
    Point2D() : x(0), y(0) {}
    Point2D(double xv, double yv) : x(xv), y(yv) {}
};

vector<Point2D> GenPointsPoly(int n, double c0, double c1, double c2) {
    vector<Point2D> pts(n);
    for (int i = 0; i < n; i++) {
        double x = dist_u(gen);
        double y = c0 + c1 * x + c2 * x * x + dist_noise(gen);
        pts[i] = { x, y };
    }
    return pts;
}


void delete_matrix(double** mat, int rows) {
    if (!mat) return;
    for (int i = 0; i < rows; ++i) delete[] mat[i];
    delete[] mat;
}

VectorXd simple(const MatrixXd& A, const VectorXd& b) {
    MatrixXd AT = A.transpose();
    MatrixXd ATA = AT * A;
    MatrixXd ATA_inv = ATA.inverse();
    VectorXd x = ATA_inv * AT * b;
    return x;
}

void qr(double** A, int N, int m,
        double** Q1, double** R1) {
    double** R = new double*[N];
    for (int i = 0; i < N; i++) {
        R[i] = new double[m];
        for (int j = 0; j < m; j++) R[i][j] = A[i][j];
    }

    vector<vector<double>> us;

    for (int k = 0; k < m && k < N - 1; k++) {
        double norm = 0.0;
        for (int i = k; i < N; i++) norm += R[i][k] * R[i][k];
        norm = sqrt(norm);
        if (norm < 1e-12) { us.push_back({}); continue; }

        double x1 = R[k][k];
        double sgn = (x1 >= 0) ? 1.0 : -1.0;
        double alpha = -sgn * norm;

        vector<double> v(N, 0.0);
        v[k] = x1 - alpha;
        for (int i = k + 1; i < N; i++) v[i] = R[i][k];

        double v_norm = 0.0;
        for (int i = k; i < N; i++) v_norm += v[i] * v[i];
        v_norm = sqrt(v_norm);
        if (v_norm < 1e-12) { us.push_back({}); continue; }

        vector<double> u(N, 0.0);
        for (int i = k; i < N; i++) u[i] = v[i] / v_norm;
        us.push_back(u);

        for (int j = k; j < m; j++) {
            double dot = 0.0;
            for (int i = k; i < N; i++) dot += u[i] * R[i][j];
            dot *= 2.0;
            for (int i = k; i < N; i++) R[i][j] -= dot * u[i];
        }
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < m; j++)
            Q1[i][j] = (i == j) ? 1.0 : 0.0;

    for (int k = (int)us.size() - 1; k >= 0; k--) {
        if (us[k].empty()) continue;
        const auto& u = us[k];
        for (int j = 0; j < m; j++) {
            double dot = 0.0;
            for (int i = k; i < N; i++) dot += u[i] * Q1[i][j];
            dot *= 2.0;
            for (int i = k; i < N; i++) Q1[i][j] -= dot * u[i];
        }
    }

    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            R1[i][j] = R[i][j];

    for (int i = 0; i < N; i++) delete[] R[i];
    delete[] R;
}

VectorXd Qr(const MatrixXd& A, const VectorXd& b) {
    int N = A.rows();
    int m = A.cols();

    double** Am = new double*[N];
    for (int i = 0; i < N; i++) {
        Am[i] = new double[m];
        for (int j = 0; j < m; j++)
            Am[i][j] = A(i, j);
    }

    double** Q1 = new double*[N];
    for (int i = 0; i < N; i++) Q1[i] = new double[m];
    double** R1 = new double*[m];
    for (int i = 0; i < m; i++) R1[i] = new double[m];

    qr(Am, N, m, Q1, R1);

    VectorXd c(m);
    for (int i = 0; i < m; i++) {
        double s = 0.0;
        for (int k = 0; k < N; k++)
            s += Q1[k][i] * b(k);
        c(i) = s;
    }

    VectorXd x(m);
    for (int i = m - 1; i >= 0; i--) {
        double s = c(i);
        for (int j = i + 1; j < m; j++)
            s -= R1[i][j] * x(j);
        x(i) = s / R1[i][i];
    }

    delete_matrix(Am, N);
    delete_matrix(Q1, N);
    delete_matrix(R1, m);

    return x;
}


VectorXd svd(const MatrixXd& A, const VectorXd& b) {
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    VectorXd x = svd.solve(b);
    return x;
}

void test(const string& name,
               const MatrixXd& A,
               const VectorXd& b,
               const VectorXd& x)
{
    cout << "\n";
    cout << "" << name << "\n";

    cout << "A (" << A.rows() << " x " << A.cols() << "):\n" << A << "\n\n";
    cout << "b (" << b.rows() << "):\n" << b.transpose() << "\n\n";

    if (x.size() > 0)
        cout << "x: " << x.transpose() << "\n\n";

    MatrixXd ATA = A.transpose() * A;
    JacobiSVD<MatrixXd> svd_cond(ATA);
    auto sv = svd_cond.singularValues();
    double cond = sv(0) / sv(sv.size() - 1);
    cout << "cond(A^T A) = " << cond << "\n\n";

    auto t1 = chrono::high_resolution_clock::now();
    VectorXd x1;
    bool ok1 = true;
    try {
        x1 = simple(A, b);
        if (!x1.allFinite()) ok1 = false;
    } catch (...) { ok1 = false; }
    auto t2 = chrono::high_resolution_clock::now();
    double ts1 = chrono::duration<double, milli>(t2 - t1).count();

    t1 = chrono::high_resolution_clock::now();
    VectorXd x2;
    bool ok2 = true;
    try {
        x2 = Qr(A, b);
        if (!x2.allFinite()) ok2 = false;
    } catch (...) { ok2 = false; }
    t2 = chrono::high_resolution_clock::now();
    double ts2 = chrono::duration<double, milli>(t2 - t1).count();

    t1 = chrono::high_resolution_clock::now();
    VectorXd x3;
    bool ok3 = true;
    try {
        x3 = svd(A, b);
        if (!x3.allFinite()) ok3 = false;
    } catch (...) { ok3 = false; }
    t2 = chrono::high_resolution_clock::now();
    double ts3 = chrono::duration<double, milli>(t2 - t1).count();

    cout << "Simple: ";
    {
        cout << "x = " << x1.transpose();
        if (x.size() > 0) cout << "   err = " << (x1 - x).norm();
        cout << "   time = " << ts1 << " ms\n";
    }
    cout << "QR:     ";
    {
        cout << "x = " << x2.transpose();
        if (x.size() > 0) cout << "   err = " << (x2 - x).norm();
        cout << "   time = " << ts2 << " ms\n";
    }
    cout << "SVD:    ";
    {
        cout << "x = " << x3.transpose();
        if (x.size() > 0) cout << "   err = " << (x3 - x).norm();
        cout << "   time = " << ts3 << " ms\n";
    }

    if (ok1 && ok2 && ok3) {
        cout << "\nРасстояния между решениями:\n";
        cout << "  ||x_simple - x_qr||  = " << (x1 - x2).norm() << "\n";
        cout << "  ||x_simple - x_svd|| = " << (x1 - x3).norm() << "\n";
        cout << "  ||x_qr - x_svd||     = " << (x2 - x3).norm() << "\n";
    }
}


int main() {
    cout << fixed << setprecision(8);
    // Хорошая A
    {
        MatrixXd A(5, 3);
        A << 1, 1, 1, 1, 2, 4, 1, 3, 9, 1, 4, 16, 1, 5, 25;
        VectorXd b(5); b << 3, 7, 13, 21, 31;
        VectorXd x(3); x << 1, 1, 1;
        test("1. Хорошая A (Вандермонд)", A, b, x);
    }

    // Почти линейно зависимые столбцы
    {
        MatrixXd A(5, 2);
        A << 1, 1,
             1, 1 + 1e-8,
             1, 1 + 2e-8,
             1, 1 + 3e-8,
             1, 1 + 4e-8;
        VectorXd b(5);
        b << 2.0, 2.00000002, 2.00000004, 2.00000006, 2.00000008;
        VectorXd x(2); x << 1, 1;
        test("2. Почти линейно зависимые столбцы", A, b, x);
    }

    // Вырожденная A
    {
        MatrixXd A(4, 2);
        A << 1, 2, 2, 4, 3, 6, 4, 8;
        VectorXd b(4); b << 3, 6, 9, 12;
        VectorXd x(2); x << 1, 1;
        test("3. Вырожденная A (ранг 1)", A, b, x);
    }

    // Большой разброс значений
    {
        MatrixXd A(4, 3);
        A << 1e8, 0, 0,
             0, 1, 0,
             0, 0, 1e-8,
             1, 1, 1;
        VectorXd b(4); b << 1e8, 1, 1e-8, 3;
        VectorXd x(3); x << 1, 1, 1;
        test("4. Большой разброс значений", A, b, x);
    }

    // Почти вырожденная A
    {
        MatrixXd A(4, 2);
        A << 1, 1,
             1, 1 + 1e-12,
             1, 1 + 2e-12,
             1, 1 + 3e-12;
        VectorXd b(4);
        b << 2.0, 2.0 + 1e-12, 2.0 + 2e-12, 2.0 + 3e-12;
        VectorXd x(2); x << 1, 1;
        test("5. Почти вырожденная A", A, b, x);
    }

    // Разреженная A
    {
        MatrixXd A(4, 4);
        A << 1, 0, 0, 1e-6,
             0, 1, 0, 1e-6,
             0, 0, 1, 1e-6,
             0, 0, 0, 1e-6;
        VectorXd b(4); b << 1, 1, 1, 1e-6;
        VectorXd x(4); x << 1, 1, 1, 1;
        test("6. Разреженная A", A, b, x);
    }

    int n = 100;
    double c0_true = 1.0, c1_true = 2.0, c2_true = 0.5;
    normal_distribution<double> noise_pts(0.0, 2.0);

    vector<Point2D> pts(n);
    for (int i = 0; i < n; i++) {
        double x = dist_u(gen);
        double y = c0_true + c1_true * x + c2_true * x * x + noise_pts(gen);
        pts[i] = { x, y };
    }

    MatrixXd A(n, 3);
    VectorXd b(n);
    for (int i = 0; i < n; i++) {
        double x = pts[i].x;
        A(i, 0) = 1.0;
        A(i, 1) = x;
        A(i, 2) = x * x;
        b(i) = pts[i].y;
    }

    MatrixXd ATA = A.transpose() * A;
    JacobiSVD<MatrixXd> svd_cond(ATA);
    auto sv = svd_cond.singularValues();
    cout << "cond(A^T A) = " << sv(0) / sv(2) << "\n\n";

    VectorXd x1 = simple(A, b);
    VectorXd x2 = Qr(A, b);
    VectorXd x3 = svd(A, b);

    cout << "Simple: c0=" << x1(0) << "  c1=" << x1(1) << "  c2=" << x1(2) << "\n";
    cout << "QR:     c0=" << x2(0) << "  c1=" << x2(1) << "  c2=" << x2(2) << "\n";
    cout << "SVD:    c0=" << x3(0) << "  c1=" << x3(1) << "  c2=" << x3(2) << "\n";
    cout << "True:   c0=" << c0_true << "  c1=" << c1_true << "  c2=" << c2_true << "\n";

    VectorXd xt(3); xt << c0_true, c1_true, c2_true;
    cout << "||x_simple - x|| = " << (x1 - xt).norm() << "\n";
    cout << "||x_qr     - x|| = " << (x2 - xt).norm() << "\n";
    cout << "||x_svd    - x|| = " << (x3 - xt).norm() << "\n";

    ofstream fpts("points.txt");
    for (auto& p : pts) fpts << p.x << " " << p.y << "\n";
    fpts.close();

    ofstream fp("poly.txt");
    double x_min = -5.0, x_max = 5.0;
    int M = 200;
    for (int i = 0; i < M; i++) {
        double x = x_min + (x_max - x_min) * i / (M - 1);
        double y_true   = c0_true + c1_true * x + c2_true * x * x;
        double y_simple = x1(0) + x1(1) * x + x1(2) * x * x;
        double y_qr     = x2(0) + x2(1) * x + x2(2) * x * x;
        double y_svd    = x3(0) + x3(1) * x + x3(2) * x * x;
        fp << x << " " << y_true << " " << y_simple << " " << y_qr << " " << y_svd << "\n";
    }
    fp.close();
    //system("gnuplot -persist -c plot.gp points.txt poly.txt");

    return 0;
}

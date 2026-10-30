#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <limits>
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

MatrixXd makeMatrix200x200(int N, double cond = 100.0) {
    MatrixXd U = MatrixXd::Random(N, N);
    MatrixXd V = MatrixXd::Random(N, N);
    
    HouseholderQR<MatrixXd> qrU(U), qrV(V);
    U = qrU.householderQ() * MatrixXd::Identity(N, N);
    V = qrV.householderQ() * MatrixXd::Identity(N, N);

    VectorXd s(N);
    for (int i = 0; i < N; i++)
        s(i) = pow(cond, -double(i) / (N - 1));

    return U * s.asDiagonal() * V.transpose();
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

VectorXd simple_tikhonov(const MatrixXd& A, const VectorXd& b, double alpha) {
    MatrixXd AT = A.transpose();
    MatrixXd ATA = AT * A;
    MatrixXd reg = alpha * MatrixXd::Identity(A.cols(), A.cols());
    MatrixXd ATA_inv = (ATA + reg).inverse();
    return ATA_inv * AT * b;
}

void qr(double** A, int N, int m,
        double** Q1, double** R1) {
    double** R = new double*[N];
    for (int i = 0; i < N; i++) {
        R[i] = new double[m];
        for (int j = 0; j < m; j++) R[i][j] = A[i][j];
    }

    vector<vector<double>> us;

    for (int k = 0; k < min(N, m); k++) {
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
          const VectorXd& x_true = VectorXd(), bool use_tikh = false)
{
    cout << "\n" << name << "\n";

    auto svA = JacobiSVD<MatrixXd>(A).singularValues();
    cout << "cond(A)=" << svA(0) / svA(svA.size()-1);

    MatrixXd ATA = A.transpose() * A;
    auto svATA = JacobiSVD<MatrixXd>(ATA).singularValues();
    cout << "  cond(A^T A)=";
    if (svATA(svATA.size()-1) > 1e-300)
        cout << svATA(0) / svATA(svATA.size()-1);
    else
        cout << "inf";
    cout << "\n";

    bool has_true = (x_true.size() > 0);

    auto run = [&](auto fn, const char* label) {
        VectorXd x;
        bool ok = true;
        auto t1 = chrono::high_resolution_clock::now();
        try { x = fn(A, b); if (!x.allFinite()) ok = false; }
        catch (...) { ok = false; }
        auto t2 = chrono::high_resolution_clock::now();
        double ts = chrono::duration<double, milli>(t2 - t1).count();

        cout << label << ": ";
        if (!ok) { cout << "FAIL\n"; return make_pair(VectorXd(), -1.0); }

        double grad_norm = (A.transpose() * (A * x - b)).norm();

        cout << "res=" << (A*x - b).norm()
             << " |grad|=" << scientific << grad_norm << fixed
             << " |x|=" << x.norm()
             << " t=" << ts << "ms";
        
        if (has_true) {
            cout << " err=" << (x - x_true).norm();
        }
        cout << "\n";
        return make_pair(x, ts);
    };

    auto r1 = run(simple, "simple");
    auto r2 = run(Qr,     "qr    ");
    auto r3 = run(svd,    "svd   ");
    if (use_tikh) {
        auto r4 = run([&](const MatrixXd& A, const VectorXd& b){ 
            return simple_tikhonov(A, b, 1e-8); 
        }, "tikh  ");
    }

    if (r1.second >= 0 && r2.second >= 0 && r3.second >= 0) {
        cout << "dist: s-qr=" << (r1.first - r2.first).norm()
             << " s-svd="  << (r1.first - r3.first).norm()
             << " qr-svd=" << (r2.first - r3.first).norm() << "\n";
    }
}

int main() {
    cout << fixed << setprecision(6);

    // 1
    {
        MatrixXd A(5,3);
        A << 1,1,1, 1,2,4, 1,3,9, 1,4,16, 1,5,25;
        VectorXd b(5); b << 3,7,13,21,31;
        VectorXd x(3); x << 1,1,1;
        test("1 vandermond", A, b, x);
    }
    // 2
    {
        MatrixXd A(5,2);
        A << 1, 1, 
             1, 1 + 1e-8, 
             1, 1 + 2e-8, 
             1, 1 + 3e-8, 
             1, 1 + 4e-8;
             
        VectorXd b(5); b << 2.0, 2.0 + 2e-8, 2.0 + 4e-8, 2.0 + 6e-8, 2.0 + 8e-8;
        
        VectorXd x(2); x << 0, 2;
        test("2 near lin dep", A, b, x, true);
    }
    // 3
    {
        MatrixXd A(4,2);
        A << 1,2, 2,4, 3,6, 4,8;
        VectorXd b(4); b << 3,6,9,12;
        test("3 singular rank1", A, b, VectorXd(), true);
    }
    // 4
    {
        MatrixXd A(4,3);
        A << 1e7, 0, 0, 
             0, 1, 0, 
             0, 0, 1e-7, 
             1, 1, 1;
        VectorXd b(4); b << 1e5, 1, 1e-7, 2.01;
        VectorXd x(3); x << 0.01, 1, 1;
        test("4 big spread", A, b, x);
    }
    // 5
    {
        MatrixXd A(4,2);
        A << 1,1, 1,1+1e-12, 1,1+2e-12, 1,1+3e-12;
        VectorXd b(4); b << 2,2+1e-12,2+2e-12,2+3e-12;
        VectorXd x(2); x << 1,1;
        test("5 near singular", A, b, x, true);
    }
    // 6
    {
        MatrixXd A(4,4);
        A << 1,0,0,1e-6, 0,1,0,1e-6, 0,0,1,1e-6, 0,0,0,1e-6;
        VectorXd b(4); b << 1,1,1,1e-6;
        VectorXd x(4); x << 1-1e-6,1-1e-6,1-1e-6,1;
        test("6 sparse", A, b, x);
    }
    // 7
    {
        int N = 200;
        MatrixXd A = makeMatrix200x200(N, 100.0); 
        VectorXd x_true = VectorXd::Random(N); 
        VectorXd b = A * x_true; 
        test("7 matrix 200x200", A, b, x_true);
    }

    // poly
    int n = 100;
    double c0 = 1.0, c1 = 2.0, c2 = 0.5;
    auto pts = GenPointsPoly(n, c0, c1, c2);

    MatrixXd A(n,3);
    VectorXd b(n);
    for (int i = 0; i < n; i++) {
        double x = pts[i].x;
        A(i,0)=1; A(i,1)=x; A(i,2)=x*x;
        b(i)=pts[i].y;
    }

    auto sv = JacobiSVD<MatrixXd>(A).singularValues();
    cout << "\npoly n=" << n << "\n";
    cout << "cond(A)=" << sv(0)/sv(2) << "\n";

    VectorXd x1 = simple(A,b), x2 = Qr(A,b), x3 = svd(A,b);
    VectorXd xt(3); xt << c0,c1,c2;

    auto print_poly = [&](const VectorXd& x, const string& name) {
        auto t1 = chrono::high_resolution_clock::now();
        VectorXd x_res = x;
        double res = (A * x - b).norm();
        double grad = (A.transpose() * (A * x - b)).norm();
        double err = (x - xt).norm();
        double norm_x = x.norm();
        auto t2 = chrono::high_resolution_clock::now();
        double ts = chrono::duration<double, milli>(t2 - t1).count();

        cout << name << ": " 
             << x.transpose() 
             << " | res=" << res 
             << " | |grad|=" << scientific << grad << fixed 
             << " | |x|=" << norm_x
             << " | t=" << ts << "ms"
             << " | err=" << err << "\n";
    };

    print_poly(x1, "simple");
    print_poly(x2, "qr    ");
    print_poly(x3, "svd   ");
    cout << "true  : " << xt.transpose() << "\n";

    ofstream fpts("points.txt");
    for (auto& p : pts) fpts << p.x << " " << p.y << "\n";
    fpts.close();

    ofstream fp("poly.txt");
    for (int i = 0; i < 200; i++) {
        double x = -5.0 + 10.0*i/199;
        fp << x
           << " " << c0+c1*x+c2*x*x
           << " " << x1(0)+x1(1)*x+x1(2)*x*x
           << " " << x2(0)+x2(1)*x+x2(2)*x*x
           << " " << x3(0)+x3(1)*x+x3(2)*x*x
           << "\n";
    }
    fp.close();

    return 0;
}

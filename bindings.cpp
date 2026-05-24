#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "engine.hpp"

namespace py = pybind11;

// Wraps Engine for Python
// Usage:
//   import cuda_search
//   engine = cuda_search.Engine(embeddings)             # embeddings: np.ndarray [N, D] float32
//   indices, scores = engine.search(query, k)           # global kernel (default)
//   indices, scores = engine.search(query, k, True)     # shared memory kernel

class PyEngine {
public:
    PyEngine(py::array_t<float> db) {
        auto buf = db.request();
        if (buf.ndim != 2)
            throw std::runtime_error("db must be 2D array [N, D]");
            
        int N = buf.shape[0];
        int D = buf.shape[1];
        engine_ = std::make_unique<Engine>(
            static_cast<float*>(buf.ptr), N, D
        );
    }

    std::pair<std::vector<int>, std::vector<float>>
    search(py::array_t<float> query, int k, bool use_tiled = false) {
        auto buf = query.request();
        if (buf.ndim != 1)
            throw std::runtime_error("query must be 1D array [D]");

        std::vector<int>   indices(k);
        std::vector<float> scores(k);
        engine_->search(
            static_cast<float*>(buf.ptr), k, use_tiled,
            indices.data(), scores.data()
        );
        return { indices, scores };
    }

    int N() const { return engine_->N(); }
    int D() const { return engine_->D(); }

private:
    std::unique_ptr<Engine> engine_;
};

PYBIND11_MODULE(cuda_search, m) {
    m.doc() = "CUDA vector search engine";

    py::class_<PyEngine>(m, "Engine")
        .def(py::init<py::array_t<float>>(),
             py::arg("db"),
             "Load embedding database onto GPU.\n"
             "db: numpy float32 array of shape [N, D]")
        .def("search", &PyEngine::search,
             py::arg("query"), py::arg("k") = 10, py::arg("use_tiled") = false,
             "Find top-K nearest neighbors.\n"
             "use_tiled=False: global kernel (default)\n"
             "use_tiled=True:  shared memory kernel\n"
             "Returns (indices, scores) tuple.")
        .def_property_readonly("N", &PyEngine::N)
        .def_property_readonly("D", &PyEngine::D);
}
from mlcourse import data_dir, lectures_dir, project_root


def test_project_paths_resolve() -> None:
    root = project_root()

    assert root.name == "machine_learning_course_basics"
    assert lectures_dir().parent == root
    assert data_dir().parent == root

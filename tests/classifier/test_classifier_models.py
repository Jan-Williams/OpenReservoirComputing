import jax
import jax.numpy as jnp
import pytest

import orc
import orc.classifier


@pytest.fixture
def dummy_classification_data():
    """Generate dummy classification data with 3 classes of sinusoidal sequences."""
    data_dim = 4
    n_classes = 3
    seq_len = 100
    n_samples_per_class = 10

    key = jax.random.PRNGKey(42)

    seqs = []
    labels = []
    for class_idx in range(n_classes):
        freq = (class_idx + 1) * 0.5  # Different frequency per class
        for _ in range(n_samples_per_class):
            key, subkey = jax.random.split(key)
            t = jnp.linspace(0, 2 * jnp.pi, seq_len).reshape(-1, 1)
            noise = 0.01 * jax.random.normal(subkey, (seq_len, data_dim))
            seq = jnp.sin(freq * t * jnp.arange(1, data_dim + 1)) + noise
            seqs.append(seq)
            labels.append(class_idx)

    train_seqs = jnp.stack(seqs)  # (n_samples, seq_len, data_dim)
    labels = jnp.array(labels, dtype=jnp.int32)
    return data_dim, n_classes, train_seqs, labels


####################### ESN CLASSIFIER DIMENSION TESTS #####################


def test_esn_classifier_initialization():
    """Test that ESNClassifier initializes with correct dimensions."""
    data_dim = 4
    n_classes = 3
    res_dim = 100

    classifier = orc.classifier.ESNClassifier(
        data_dim=data_dim, n_classes=n_classes, res_dim=res_dim, seed=0
    )

    assert classifier.data_dim == data_dim
    assert classifier.n_classes == n_classes
    assert classifier.res_dim == res_dim
    assert classifier.in_dim == data_dim
    assert classifier.out_dim == n_classes
    assert classifier.state_repr == "final"
    assert classifier.embedding.in_dim == data_dim


def test_esn_classifier_force_shapes():
    """Test that force method produces correct output shapes."""
    data_dim = 4
    n_classes = 3
    res_dim = 100
    seq_len = 50

    classifier = orc.classifier.ESNClassifier(
        data_dim=data_dim, n_classes=n_classes, res_dim=res_dim, seed=0
    )

    in_seq = jax.random.normal(jax.random.PRNGKey(0), (seq_len, data_dim))
    res_state = jnp.zeros(res_dim, dtype=jnp.float64)

    res_seq = classifier.force(in_seq, res_state)

    assert res_seq.shape == (seq_len, res_dim)


def test_esn_classifier_classify_shapes():
    """Test that classify method produces correct output shapes."""
    data_dim = 4
    n_classes = 3
    res_dim = 100
    seq_len = 50

    classifier = orc.classifier.ESNClassifier(
        data_dim=data_dim, n_classes=n_classes, res_dim=res_dim, seed=0
    )

    in_seq = jax.random.normal(jax.random.PRNGKey(0), (seq_len, data_dim))
    res_state = jnp.zeros(res_dim, dtype=jnp.float64)

    probs = classifier.classify(in_seq, res_state)

    assert probs.shape == (n_classes,)
    # Probabilities should sum to 1
    assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)
    # All probabilities should be non-negative
    assert jnp.all(probs >= 0)


def test_esn_classifier_classify_default_state():
    """Test that classify works with default (None) reservoir state."""
    data_dim = 4
    n_classes = 3
    res_dim = 100
    seq_len = 50

    classifier = orc.classifier.ESNClassifier(
        data_dim=data_dim, n_classes=n_classes, res_dim=res_dim, seed=0
    )

    in_seq = jax.random.normal(jax.random.PRNGKey(0), (seq_len, data_dim))

    probs = classifier.classify(in_seq)

    assert probs.shape == (n_classes,)
    assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)


####################### ESN CLASSIFIER TRAINING TESTS #####################


def test_esn_classifier_train(dummy_classification_data):
    """Test training of ESNClassifier on dummy data."""
    data_dim, n_classes, train_seqs, labels = dummy_classification_data

    res_dim = 200

    classifier = orc.classifier.ESNClassifier(
        data_dim=data_dim,
        n_classes=n_classes,
        res_dim=res_dim,
        seed=0,
    )

    classifier_trained = orc.classifier.train_ESNClassifier(
        classifier,
        train_seqs=train_seqs,
        labels=labels,
        beta=1e-6,
    )

    assert classifier_trained is not None

    # Verify trained model can classify and produces valid probabilities
    probs = classifier_trained.classify(train_seqs[0])
    assert probs.shape == (n_classes,)
    assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)
    assert jnp.all(jnp.isfinite(probs))


def test_esn_classifier_mean_state(dummy_classification_data):
    """Test classifier with mean state representation."""
    data_dim, n_classes, train_seqs, labels = dummy_classification_data

    res_dim = 200

    classifier = orc.classifier.ESNClassifier(
        data_dim=data_dim,
        n_classes=n_classes,
        res_dim=res_dim,
        seed=0,
        state_repr="mean",
    )

    classifier_trained = orc.classifier.train_ESNClassifier(
        classifier,
        train_seqs=train_seqs,
        labels=labels,
        spinup=10,
        beta=1e-6,
    )

    assert classifier_trained is not None

    # Verify trained model can classify
    probs = classifier_trained.classify(train_seqs[0], spinup=10)
    assert probs.shape == (n_classes,)
    assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)
    assert jnp.all(jnp.isfinite(probs))


def test_esn_classifier_invalid_state_repr():
    """Test that invalid state_repr raises an error."""
    with pytest.raises(ValueError, match="state_repr must be"):
        orc.classifier.ESNClassifier(
            data_dim=4, n_classes=3, res_dim=100, state_repr="invalid"
        )


####################### UNIFIED train_RCClassifier TESTS #####################


def test_train_rcclassifier_esn(dummy_classification_data):
    """Test that train_RCClassifier gives same results to train_ESNClassifier."""
    data_dim, n_classes, train_seqs, labels = dummy_classification_data

    res_dim = 200

    classifier = orc.classifier.ESNClassifier(
        data_dim=data_dim,
        n_classes=n_classes,
        res_dim=res_dim,
        seed=0,
    )

    # Train with both functions
    cls_old = orc.classifier.train_ESNClassifier(
        classifier,
        train_seqs=train_seqs,
        labels=labels,
        beta=1e-6,
    )

    cls_new = orc.classifier.train_RCClassifier(
        classifier,
        train_seqs=train_seqs,
        labels=labels,
        beta=1e-6,
    )

    # Verify identical results
    assert jnp.allclose(cls_old.readout.wout, cls_new.readout.wout)


def test_train_rcclassifier_mean_state(dummy_classification_data):
    """Test train_RCClassifier with mean state representation."""
    data_dim, n_classes, train_seqs, labels = dummy_classification_data

    res_dim = 200

    classifier = orc.classifier.ESNClassifier(
        data_dim=data_dim,
        n_classes=n_classes,
        res_dim=res_dim,
        seed=0,
        state_repr="mean",
    )

    cls_trained = orc.classifier.train_RCClassifier(
        classifier,
        train_seqs=train_seqs,
        labels=labels,
        spinup=10,
        beta=1e-6,
    )

    assert cls_trained is not None
    probs = cls_trained.classify(train_seqs[0], spinup=10)
    assert probs.shape == (n_classes,)
    assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)
    assert jnp.all(jnp.isfinite(probs))


def test_train_rcclassifier_quadratic(dummy_classification_data):
    """Test train_RCClassifier with quadratic readout."""
    data_dim, n_classes, train_seqs, labels = dummy_classification_data

    res_dim = 200

    classifier = orc.classifier.ESNClassifier(
        data_dim=data_dim,
        n_classes=n_classes,
        res_dim=res_dim,
        seed=0,
        quadratic=True,
    )

    # Train with both functions and verify identical results
    cls_old = orc.classifier.train_ESNClassifier(
        classifier,
        train_seqs=train_seqs,
        labels=labels,
        beta=1e-6,
    )

    cls_new = orc.classifier.train_RCClassifier(
        classifier,
        train_seqs=train_seqs,
        labels=labels,
        beta=1e-6,
    )

    assert jnp.allclose(cls_old.readout.wout, cls_new.readout.wout)

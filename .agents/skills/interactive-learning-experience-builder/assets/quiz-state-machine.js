(function exposeQuizStateMachine(root, factory) {
  "use strict";

  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.LearningExperienceQuiz = api;
})(typeof globalThis === "undefined" ? this : globalThis, function createQuizStateMachine() {
  "use strict";

  function blankQuestionState() {
    return {
      attempts: 0,
      firstAttemptCorrect: null,
      complete: false,
    };
  }

  function createQuizState(questionCount, settings = {}) {
    return {
      quizIndex: 0,
      questionStates: Array.from({ length: questionCount }, blankQuestionState),
      responses: [],
      selection: [],
      feedback: null,
      showResults: false,
      settings: { ...settings },
    };
  }

  function currentQuestionState(state) {
    return state.questionStates[state.quizIndex];
  }

  function normalizedAnswer(value) {
    const values = Array.isArray(value) ? value : [value];
    return values.map((item) => String(item).trim().toLocaleLowerCase()).sort();
  }

  function answersMatch(selected, answer) {
    return JSON.stringify(normalizedAnswer(selected)) ===
      JSON.stringify(normalizedAnswer(answer));
  }

  function selectAnswer(state, selected) {
    if (currentQuestionState(state)?.complete) return state;
    const values = Array.isArray(selected) ? selected : [selected];
    return {
      ...state,
      selection: values.map((value) => String(value).trim()).filter(Boolean),
      feedback: null,
    };
  }

  function submitAnswer(state, question) {
    const questionState = currentQuestionState(state);
    if (!questionState || questionState.complete) return state;
    if (!state.selection.length) {
      return {
        ...state,
        feedback: {
          correct: false,
          message: "Choose an answer before checking.",
          missing: true,
        },
      };
    }

    const correct = answersMatch(state.selection, question.answer);
    const attempts = questionState.attempts + 1;
    const updatedQuestionState = {
      attempts,
      firstAttemptCorrect:
        questionState.firstAttemptCorrect === null
          ? correct
          : questionState.firstAttemptCorrect,
      complete: correct,
    };
    const questionStates = state.questionStates.map((item, index) =>
      index === state.quizIndex ? updatedQuestionState : item
    );
    const feedback = {
      correct,
      message: question.explanation,
      missing: false,
    };

    if (!correct) {
      return { ...state, questionStates, feedback };
    }

    return {
      ...state,
      questionStates,
      feedback,
      responses: [
        ...state.responses,
        {
          id: question.id,
          prompt: question.prompt,
          selected: [...state.selection],
          answer: question.answer,
          explanation: question.explanation,
          correct,
          attempts,
          firstAttemptCorrect: updatedQuestionState.firstAttemptCorrect,
        },
      ],
    };
  }

  function canAdvance(state) {
    return !state.showResults && Boolean(currentQuestionState(state)?.complete);
  }

  function completedCount(state) {
    return state.questionStates.filter((questionState) => questionState.complete).length;
  }

  function nextQuestion(state) {
    if (!canAdvance(state)) return state;
    if (state.quizIndex >= state.questionStates.length - 1) {
      return {
        ...state,
        selection: [],
        feedback: null,
        showResults: true,
      };
    }
    return {
      ...state,
      quizIndex: state.quizIndex + 1,
      selection: [],
      feedback: null,
    };
  }

  function summarize(state) {
    return {
      completed: completedCount(state),
      firstAttemptCorrect: state.responses.filter(
        (response) => response.firstAttemptCorrect
      ).length,
      totalAttempts: state.responses.reduce(
        (total, response) => total + response.attempts,
        0
      ),
    };
  }

  function withSettings(state, settings) {
    return { ...state, settings: { ...settings } };
  }

  function retryQuiz(state) {
    return createQuizState(state.questionStates.length, state.settings);
  }

  return {
    answersMatch,
    canAdvance,
    completedCount,
    createQuizState,
    currentQuestionState,
    nextQuestion,
    retryQuiz,
    selectAnswer,
    submitAnswer,
    summarize,
    withSettings,
  };
});

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const QuizStateMachine = require(
  "../.agents/skills/interactive-learning-experience-builder/assets/quiz-state-machine.js",
);

const question = {
  id: "question-1",
  prompt: "Which answer is correct?",
  answer: "Correct",
  explanation: "The source supports Correct.",
};

test("wrong answers stay incomplete and changing selection clears feedback", () => {
  let state = QuizStateMachine.createQuizState(1, { difficulty: "foundations" });
  state = QuizStateMachine.selectAnswer(state, ["Wrong"]);
  state = QuizStateMachine.submitAnswer(state, question);

  assert.equal(state.questionStates[0].attempts, 1);
  assert.equal(state.questionStates[0].complete, false);
  assert.equal(QuizStateMachine.canAdvance(state), false);
  assert.equal(QuizStateMachine.completedCount(state), 0);
  assert.equal(state.feedback.correct, false);

  state = QuizStateMachine.selectAnswer(state, ["Still wrong"]);
  assert.equal(state.feedback, null);
  assert.equal(QuizStateMachine.canAdvance(state), false);
});

test("only a correct answer exposes Next and increments completion progress", () => {
  let state = QuizStateMachine.createQuizState(1, { difficulty: "applied" });
  state = QuizStateMachine.selectAnswer(state, ["Wrong"]);
  state = QuizStateMachine.submitAnswer(state, question);

  assert.equal(QuizStateMachine.canAdvance(state), false);
  assert.equal(QuizStateMachine.completedCount(state), 0);

  state = QuizStateMachine.selectAnswer(state, ["Correct"]);
  state = QuizStateMachine.submitAnswer(state, question);

  assert.equal(QuizStateMachine.canAdvance(state), true);
  assert.equal(QuizStateMachine.completedCount(state), 1);
  assert.equal(state.questionStates[0].complete, true);
});

test("results preserve first-attempt accuracy and total attempts", () => {
  const secondQuestion = { ...question, id: "question-2" };
  let state = QuizStateMachine.createQuizState(2, { difficulty: "challenge" });

  state = QuizStateMachine.selectAnswer(state, ["Wrong"]);
  state = QuizStateMachine.submitAnswer(state, question);
  state = QuizStateMachine.selectAnswer(state, ["Correct"]);
  state = QuizStateMachine.submitAnswer(state, question);
  state = QuizStateMachine.nextQuestion(state);
  state = QuizStateMachine.selectAnswer(state, ["Correct"]);
  state = QuizStateMachine.submitAnswer(state, secondQuestion);
  state = QuizStateMachine.nextQuestion(state);

  assert.equal(state.showResults, true);
  assert.deepEqual(QuizStateMachine.summarize(state), {
    completed: 2,
    firstAttemptCorrect: 1,
    totalAttempts: 3,
  });
});

test("Retry resets every quiz field while preserving learner settings", () => {
  const settings = {
    difficulty: "challenge",
    focus: true,
    colorBlind: true,
    breakPrompts: false,
  };
  let state = QuizStateMachine.createQuizState(1, settings);
  state = QuizStateMachine.selectAnswer(state, ["Wrong"]);
  state = QuizStateMachine.submitAnswer(state, question);
  state = QuizStateMachine.selectAnswer(state, ["Correct"]);
  state = QuizStateMachine.submitAnswer(state, question);
  state = QuizStateMachine.nextQuestion(state);

  state = QuizStateMachine.retryQuiz(state);

  assert.deepEqual(state.settings, settings);
  assert.equal(state.quizIndex, 0);
  assert.deepEqual(state.responses, []);
  assert.deepEqual(state.selection, []);
  assert.equal(state.feedback, null);
  assert.equal(state.showResults, false);
  assert.deepEqual(state.questionStates, [
    { attempts: 0, firstAttemptCorrect: null, complete: false },
  ]);
  assert.equal(QuizStateMachine.canAdvance(state), false);
  assert.equal(QuizStateMachine.completedCount(state), 0);
});

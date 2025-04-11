'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Question } from '../types/quiz';
import { questions } from '../data/quizData';
import FrequentlyMissed from './FrequentlyMissed';

interface ShuffledQuestion extends Question {
  shuffledOptions: string[];
  correctIndex: number;
}

interface QuizState {
  currentQuestionIndex: number;
  score: number;
  selectedAnswers: number[];
  showResults: boolean;
  quizQuestions: ShuffledQuestion[];
  showFeedback: boolean;
  lastAnswerCorrect: boolean;
  selectedAnswer: number | null;
}

type TabType = 'quiz' | 'history' | 'missed';

const TABS: Record<'QUIZ' | 'HISTORY' | 'MISSED', TabType> = {
  QUIZ: 'quiz',
  HISTORY: 'history',
  MISSED: 'missed'
} as const;

export const Quiz: React.FC = () => {
  const [isClient, setIsClient] = useState(false);
  const [quizState, setQuizState] = useState<QuizState>({
    currentQuestionIndex: 0,
    score: 0,
    selectedAnswers: [],
    showResults: false,
    quizQuestions: [],
    showFeedback: false,
    lastAnswerCorrect: false,
    selectedAnswer: null
  });
  const [incorrectAnswers, setIncorrectAnswers] = useState<Record<string, number>>({});
  const [activeTab, setActiveTab] = useState<TabType>(TABS.QUIZ);
  const [scoreHistory, setScoreHistory] = useState<number[]>([]);

  const startNewQuiz = useCallback(() => {
    // Shuffle questions and select first 15
    const shuffled = [...questions]
      .sort(() => Math.random() - 0.5)
      .slice(0, 15)
      .map(q => {
        const shuffledOptions = [...q.options].sort(() => Math.random() - 0.5);
        const correctIndex = shuffledOptions.indexOf(q.options[q.correctAnswer]);
        return {
          ...q,
          shuffledOptions,
          correctIndex
        };
      });
    setQuizState(prev => ({
      ...prev,
      currentQuestionIndex: 0,
      score: 0,
      selectedAnswers: [],
      showResults: false,
      quizQuestions: shuffled
    }));
  }, []);

  useEffect(() => {
    setIsClient(true);
    const savedHistory = localStorage.getItem('scoreHistory');
    if (savedHistory) {
      setScoreHistory(JSON.parse(savedHistory));
    }
    const savedIncorrect = localStorage.getItem('incorrectAnswers');
    if (savedIncorrect) {
      setIncorrectAnswers(JSON.parse(savedIncorrect));
    }
    startNewQuiz();
  }, [startNewQuiz]);

  const handleAnswerSelect = (answerIndex: number) => {
    const currentQuestion = quizState.quizQuestions[quizState.currentQuestionIndex];
    const isCorrect = answerIndex === currentQuestion.correctIndex;
    
    const newSelectedAnswers = [...quizState.selectedAnswers, answerIndex];
    const newScore = isCorrect ? quizState.score + 1 : quizState.score;
    
    if (!isCorrect) {
      setIncorrectAnswers(prev => {
        const newIncorrect = { ...prev, [currentQuestion.id.toString()]: (prev[currentQuestion.id.toString()] || 0) + 1 };
        if (isClient) {
          localStorage.setItem('incorrectAnswers', JSON.stringify(newIncorrect));
        }
        return newIncorrect;
      });
    }

    setQuizState(prev => ({
      ...prev,
      selectedAnswers: newSelectedAnswers,
      score: newScore,
      showFeedback: true,
      lastAnswerCorrect: isCorrect,
      selectedAnswer: answerIndex
    }));
  };

  const handleNextQuestion = () => {
    if (quizState.currentQuestionIndex < quizState.quizQuestions.length - 1) {
      setQuizState(prev => ({
        ...prev,
        currentQuestionIndex: prev.currentQuestionIndex + 1,
        showFeedback: false,
        selectedAnswer: null
      }));
    } else {
      const finalScore = quizState.score;
      const newScoreHistory = [...scoreHistory, finalScore];
      setScoreHistory(newScoreHistory);
      if (isClient) {
        localStorage.setItem('scoreHistory', JSON.stringify(newScoreHistory));
      }
      setQuizState(prev => ({
        ...prev,
        showResults: true,
        showFeedback: false,
        selectedAnswer: null
      }));
    }
  };

  const getFrequentlyMissedQuestions = () => {
    const missedQuestions = Object.entries(incorrectAnswers)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([id, count]) => {
        const question = questions.find(q => q.id === parseInt(id));
        return question ? { question, count } : null;
      })
      .filter((q): q is { question: Question; count: number } => q !== null);

    return missedQuestions;
  };

  if (!isClient) {
    return null;
  }

  if (activeTab === TABS.HISTORY) {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold">Score History</h2>
          <button
            onClick={() => setActiveTab(TABS.QUIZ)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Back to Quiz
          </button>
        </div>
        <div className="space-y-2">
          {scoreHistory.map((score, index) => (
            <div key={index} className="p-4 bg-white rounded-lg shadow">
              Attempt {index + 1}: {score} correct answers
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (activeTab === TABS.MISSED) {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold">Frequently Missed Problems</h2>
          <button
            onClick={() => setActiveTab(TABS.QUIZ)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Back to Quiz
          </button>
        </div>
        <FrequentlyMissed frequentlyMissed={getFrequentlyMissedQuestions()} />
      </div>
    );
  }

  if (quizState.showResults) {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <h2 className="text-2xl font-bold mb-4">Quiz Results</h2>
        <p className="text-lg mb-4">Your score: {quizState.score} out of {quizState.quizQuestions.length}</p>
        <div className="space-y-4">
          {quizState.quizQuestions.map((question, index) => (
            <div key={question.id} className="p-4 bg-white rounded-lg shadow">
              <p className="font-semibold">{question.question}</p>
              <p className={`mt-2 ${quizState.selectedAnswers[index] === question.correctIndex ? 'text-green-600' : 'text-red-600'}`}>
                Your answer: {question.shuffledOptions[quizState.selectedAnswers[index]]}
              </p>
              <p className="text-green-600">Correct answer: {question.options[question.correctAnswer]}</p>
              <p className="text-gray-600 mt-2">{question.explanation}</p>
              <a href={question.reference} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
                Learn more
              </a>
            </div>
          ))}
        </div>
        <div className="mt-6 space-x-4">
          <button
            onClick={startNewQuiz}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Try Again
          </button>
          <button
            onClick={() => setActiveTab(TABS.MISSED)}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            View Missed Problems
          </button>
          <button
            onClick={() => setActiveTab(TABS.HISTORY)}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            View History
          </button>
        </div>
      </div>
    );
  }

  const currentQuestion = quizState.quizQuestions[quizState.currentQuestionIndex];

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="mb-8">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8" aria-label="Tabs">
            <button
              onClick={() => setActiveTab(TABS.QUIZ)}
              className={`${
                activeTab === TABS.QUIZ
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              Quiz
            </button>
            <button
              onClick={() => setActiveTab(TABS.HISTORY)}
              className={`${
                activeTab === TABS.HISTORY
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              Score History
            </button>
            <button
              onClick={() => setActiveTab(TABS.MISSED)}
              className={`${
                activeTab === TABS.MISSED
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              Missed Problems
            </button>
          </nav>
        </div>
      </div>
      <div className="max-w-2xl mx-auto p-6">
        <div className="mb-4">
          <h2 className="text-2xl font-bold">Question {quizState.currentQuestionIndex + 1} of {quizState.quizQuestions.length}</h2>
          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
            <div
              className="bg-blue-600 h-2.5 rounded-full"
              style={{ width: `${((quizState.currentQuestionIndex + 1) / quizState.quizQuestions.length) * 100}%` }}
            ></div>
          </div>
        </div>
        <div className="mb-8">
          <h3 className="text-lg font-medium mb-4">{currentQuestion.question}</h3>
          <div className="space-y-3">
            {currentQuestion.shuffledOptions.map((option, index) => (
              <button
                key={index}
                onClick={() => !quizState.showFeedback && handleAnswerSelect(index)}
                disabled={quizState.showFeedback}
                className={`w-full text-left p-4 rounded-lg border transition-all ${
                  quizState.showFeedback
                    ? quizState.selectedAnswer === index
                      ? index === currentQuestion.correctIndex
                        ? 'border-green-500 bg-green-50 text-green-700'
                        : 'border-red-500 bg-red-50 text-red-700'
                      : index === currentQuestion.correctIndex
                      ? 'border-green-500 bg-green-50 text-green-700'
                      : 'border-gray-200 bg-white'
                    : 'border-gray-200 hover:border-blue-500 hover:bg-blue-50'
                }`}
              >
                {option}
                {quizState.showFeedback && quizState.selectedAnswer === index && (
                  <span className="ml-2">
                    {index === currentQuestion.correctIndex ? '✓' : '✗'}
                  </span>
                )}
              </button>
            ))}
          </div>
          {quizState.showFeedback && (
            <div className={`mt-4 p-4 rounded-lg ${
              quizState.lastAnswerCorrect ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              <p className="font-medium">
                {quizState.lastAnswerCorrect ? 'Correct!' : 'Incorrect'}
              </p>
              {!quizState.lastAnswerCorrect && (
                <p className="mt-2">
                  The correct answer is: {currentQuestion.options[currentQuestion.correctAnswer]}
                </p>
              )}
              <p className="mt-2">{currentQuestion.explanation}</p>
              {currentQuestion.reference && (
                <a
                  href={currentQuestion.reference}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 transition-colors inline-block mt-2"
                >
                  Learn more about this topic →
                </a>
              )}
              <div className="mt-4">
                <button
                  onClick={handleNextQuestion}
                  className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                >
                  {quizState.currentQuestionIndex < quizState.quizQuestions.length - 1 ? 'Next Question' : 'View Results'}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}; 
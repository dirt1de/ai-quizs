import React from 'react';
import { Question } from '../types/quiz';

interface FrequentlyMissedProps {
  frequentlyMissed: { question: Question; count: number }[];
}

const FrequentlyMissed: React.FC<FrequentlyMissedProps> = ({ frequentlyMissed }) => {
  return (
    <div>
      {frequentlyMissed.length === 0 ? (
        <p className="text-gray-500">No frequently missed problems yet.</p>
      ) : (
        <div className="space-y-6">
          {frequentlyMissed.map(({ question, count }, index) => (
            <div key={index} className="relative border border-gray-100 rounded-xl p-6 hover:shadow-sm transition-shadow">
              <div className="absolute top-4 right-4">
                <span className="inline-flex items-center justify-center px-3 py-1 text-sm font-medium text-red-700 bg-red-100 rounded-full">
                  Missed {count} {count === 1 ? 'time' : 'times'}
                </span>
              </div>
              <div className="pr-32">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">{question.question}</h3>
                <div className="mb-4">
                  <p className="font-medium text-gray-700 mb-1">Correct Answer:</p>
                  <p className="text-green-600">{question.options[question.correctAnswer]}</p>
                </div>
                <div className="mb-4">
                  <p className="font-medium text-gray-700 mb-1">Explanation:</p>
                  <p className="text-gray-600">{question.explanation}</p>
                </div>
                {question.reference && (
                  <a
                    href={question.reference}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800 transition-colors inline-block"
                  >
                    Learn more
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default FrequentlyMissed; 
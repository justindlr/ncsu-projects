LIBNAME data "C:\SAS";

DATA data.games;
	INFILE "C:\SAS\games.csv" FIRSTOBS = 2 DSD;
	INPUT GameID: Date :YYMMDD14. Tournament: $12. TournamentGame Team $ Opponent: $12. Outcome $ TeamPoints TeamPointsAllowed;
RUN;

DATA data.rpe;
	INFILE "C:\SAS\rpe.csv" FIRSTOBS = 2 DSD;
	INPUT Date :YYMMDD14. PlayerID Training $ SessionType: $16. Duration RPE SessionLoad DailyLoad	AcuteLoad ChronicLoad AcuteChronicRatio	ObjectiveRating	FocusRating	BestOutOfMyself: $10.;
RUN;

DATA data.wellness;
	INFILE "C:\SAS\wellness.csv" FIRSTOBS = 2 DSD;
	INPUT Date :YYMMDD14. PlayerID Fatigue Soreness Desire Irritability BedTime: Time14. WakeTime: Time14. SleepHours SleepQuality MonitoringScore Pain $ Illness: $12. Menstruation $ Nutrition: $9. NutritionAdjustment: $ USGMeasurement $ USG: TrainingReadiness: PERCENT3.;
	FORMAT TrainingReadiness PERCENT10.;
RUN;

DATA data.gps;
	INFILE "C:\SAS\gps.csv" FIRSTOBS = 2 DSD;
	INPUT GameID Half PlayerID FrameID Time: Time14. GameClock: Time14. Speed AccelImpulse AcceLoad AccelX AccelY AccelZ Longitude Latitude;
RUN;


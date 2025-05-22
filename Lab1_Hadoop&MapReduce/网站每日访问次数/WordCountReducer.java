import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable>{
    IntWritable v = new IntWritable();
    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        // 1. 累加求和
        int sum = 0;
        for (IntWritable count : values) {
            sum += count.get();
        }
        // 2. 输出结果，格式为 "日期  次数"
        v.set(sum);
        context.write(key, v);
    }
}
